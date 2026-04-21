import cv2
import numpy as np
import argparse
import time
import math
import os
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO


class UnattendedBagDetector:
    
    def __init__(self, model_path, max_owner_distance=150, alert_time=5, conf_threshold=0.4, stillness_frames=30):
        print(f"Loading model: {model_path}")
        
        if model_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            self.model = YOLO(model_path)
            print(f"Loaded pre-trained {model_path}")
        elif os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.max_owner_distance = max_owner_distance
        self.alert_time = alert_time
        self.conf_threshold = conf_threshold
        self.stillness_threshold = 15  # pixels - bag movement threshold
        self.stillness_frames = stillness_frames  # frames to confirm bag is stationary
        
        # 4 Classes: 0=bag, 1=luggage, 2=person, 3=person_with_bag
        self.class_names = ['bag', 'luggage', 'person', 'person_with_bag']
        self.class_colors = {
            0: (255, 0, 0),      # Blue - bag
            1: (0, 165, 255),    # Orange - luggage
            2: (0, 255, 0),      # Green - person
            3: (255, 0, 255)     # Magenta - person with bag
        }
        
        self.person_tracker = {}
        self.bag_tracker = {}
        self.bag_owners = {}  # bag_id -> person_id
        self.person_bags = defaultdict(list)  # person_id -> list of bag_ids
        
        # Motion tracking
        self.bag_position_history = {}  # Track bag movement
        self.bag_is_still = {}  # Count of frames bag is still
        self.bag_still_start_time = {}  # When bag became still
        
        # Abandonment tracking
        self.bag_abandon_start = {}  # When abandonment counter started
        self.bag_alerted = set()  # Already alerted bags
        
        self.person_position_history = {}  # Track person movement
        self.bag_colors = {}  # Unique color for each bag-owner pair
        self.person_last_seen = {}  # Track when person was last detected
        
        self.frame_count = 0
        self.detector_fps = 0
        self.last_time = time.time()
        
        print("Model loaded successfully!")
        print("\nDetection Logic:")
        print("1. Person walks with bag → No alert (person_with_bag detected)")
        print("2. Person puts bag down → No alert (bag stays with person or close)")
        print("3. Bag becomes STILL for 30 frames → Monitoring starts")
        print("4. Person walks AWAY from bag → Alert timer starts")
        print("5. Person leaves CAMERA VIEW → Bag marked as ABANDONED")
        print("6. Alert triggers after 5 seconds → RED ALERT with line to last seen owner")

    def get_distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def is_position_stable(self, current_pos, history_queue, threshold=15):
        """Check if bag is stationary"""
        if len(history_queue) < 5:
            return False
        
        avg_x = np.mean([p[0] for p in history_queue])
        avg_y = np.mean([p[1] for p in history_queue])
        
        movement = self.get_distance(current_pos, (int(avg_x), int(avg_y)))
        return movement < threshold

    def update_tracks(self, detections):
        """Update person and bag tracks from current frame"""
        # Separate detections
        persons = [d for d in detections if d['class_id'] in [2, 3]]  # person or person_with_bag
        bags = [d for d in detections if d['class_id'] in [0, 1]]      # bag or luggage
        
        # Update person tracker
        old_persons = self.person_tracker.copy()
        self.person_tracker.clear()
        
        for i, person in enumerate(persons):
            center = self.get_box_center(person['box'])
            self.person_tracker[i] = {
                'center': center,
                'box': person['box'],
                'conf': person['conf'],
                'class_id': person['class_id'],
                'class_name': self.class_names[person['class_id']]
            }
            
            # Track position history
            if i not in self.person_position_history:
                self.person_position_history[i] = deque(maxlen=20)
            self.person_position_history[i].append(center)
            
            # Mark person as seen now
            self.person_last_seen[i] = time.time()
        
        # Update bag tracker
        self.bag_tracker.clear()
        
        for i, bag in enumerate(bags):
            center = self.get_box_center(bag['box'])
            self.bag_tracker[i] = {
                'center': center,
                'box': bag['box'],
                'conf': bag['conf'],
                'class_id': bag['class_id'],
                'class_name': self.class_names[bag['class_id']]
            }
            
            # Track position history for motion detection
            if i not in self.bag_position_history:
                self.bag_position_history[i] = deque(maxlen=30)
            self.bag_position_history[i].append(center)
            
            # Check if bag is stationary
            if self.is_position_stable(center, self.bag_position_history[i], self.stillness_threshold):
                if i not in self.bag_is_still:
                    self.bag_is_still[i] = 0
                self.bag_is_still[i] += 1
            else:
                # Bag is moving - reset counter
                self.bag_is_still[i] = 0
                # Reset abandonment timer if bag is picked up and moving
                if i in self.bag_abandon_start:
                    del self.bag_abandon_start[i]
                if i in self.bag_alerted:
                    self.bag_alerted.discard(i)
        
        # Assign bags to owners
        self._assign_bags_to_persons()

    def _assign_bags_to_persons(self):
        """Link each bag to its nearest person (owner)"""
        self.person_bags.clear()
        
        for bag_id in self.bag_tracker.keys():
            bag_center = self.bag_tracker[bag_id]['center']
            
            # Find closest person
            min_dist = float('inf')
            closest_person = None
            
            for person_id in self.person_tracker.keys():
                person_center = self.person_tracker[person_id]['center']
                dist = self.get_distance(bag_center, person_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_person = person_id
            
            # Assign owner if person is close
            if closest_person is not None and min_dist < self.max_owner_distance * 2:
                self.bag_owners[bag_id] = closest_person
                self.person_bags[closest_person].append(bag_id)
                
                # Assign unique color to this bag
                if bag_id not in self.bag_colors:
                    color_idx = (closest_person + bag_id) % 10
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                             (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
                             (0, 128, 128), (255, 165, 0)]
                    self.bag_colors[bag_id] = colors[color_idx]
            elif bag_id in self.bag_owners:
                # Keep previous owner if person not found
                owner_id = self.bag_owners[bag_id]
                self.person_bags[owner_id].append(bag_id)

    def detect_abandoned_bags(self):
        """Detect abandoned bags - ONLY when person is gone from camera"""
        current_time = time.time()
        abandoned_bags = []
        
        for bag_id in self.bag_tracker.keys():
            bag_center = self.bag_tracker[bag_id]['center']
            owner_id = self.bag_owners.get(bag_id)
            bag_still_count = self.bag_is_still.get(bag_id, 0)
            
            # Stage 1: Bag must be STILL
            is_bag_still = bag_still_count >= self.stillness_frames
            
            if not is_bag_still:
                # Bag is moving - not abandoned yet
                if bag_id in self.bag_abandon_start:
                    del self.bag_abandon_start[bag_id]
                if bag_id in self.bag_alerted:
                    self.bag_alerted.discard(bag_id)
                continue
            
            # Stage 2: Owner must be GONE (far away or out of frame)
            owner_gone = False
            
            if owner_id is None:
                # No owner assigned - bag is already separated
                owner_gone = True
            elif owner_id not in self.person_tracker:
                # Owner was detected before but NOT in current frame
                # Person walked out of camera
                owner_gone = True
            else:
                # Owner is still in frame - check distance
                owner_center = self.person_tracker[owner_id]['center']
                distance = self.get_distance(bag_center, owner_center)
                
                # If person is far away (walked away)
                if distance > self.max_owner_distance:
                    owner_gone = True
            
            # Stage 3: Trigger alert ONLY if bag is still AND owner is gone
            is_abandoned = is_bag_still and owner_gone
            
            if is_abandoned:
                # Start abandonment timer
                if bag_id not in self.bag_abandon_start:
                    self.bag_abandon_start[bag_id] = current_time
                
                abandon_duration = current_time - self.bag_abandon_start[bag_id]
                
                # Alert after threshold time
                if abandon_duration > self.alert_time and bag_id not in self.bag_alerted:
                    abandoned_bags.append({
                        'bag_id': bag_id,
                        'box': self.bag_tracker[bag_id]['box'],
                        'center': bag_center,
                        'duration': abandon_duration,
                        'class_name': self.bag_tracker[bag_id]['class_name'],
                        'owner_id': owner_id
                    })
                    self.bag_alerted.add(bag_id)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] ALERT: Abandoned {self.bag_tracker[bag_id]['class_name']} detected!")
                    if owner_id is not None:
                        print(f"  Owner: Person {owner_id} (left camera)")
                    print(f"  Duration: {abandon_duration:.1f}s")
            else:
                # Bag picked up or owner returned - reset
                if bag_id in self.bag_abandon_start:
                    del self.bag_abandon_start[bag_id]
                if bag_id in self.bag_alerted:
                    self.bag_alerted.discard(bag_id)
        
        return abandoned_bags

    def process_frame(self, frame, skip_frames=1, resize=640):
        self.frame_count += 1
        
        # Skip frames for speed
        if self.frame_count % skip_frames != 0:
            return frame, []
        
        # Resize for faster processing
        h, w = frame.shape[:2]
        if w > resize:
            scale = resize / w
            new_w = resize
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            for box, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cls_id_int = int(cls_id)
                
                # Stricter filtering for bags/luggage
                if cls_id_int in [0, 1]:  # bag or luggage
                    if float(conf) < self.conf_threshold + 0.15:
                        continue
                    
                    width = x2 - x1
                    height = y2 - y1
                    if width > 400 or height > 400:
                        continue
                
                detections.append({
                    'class_id': cls_id_int,
                    'class_name': self.class_names[cls_id_int],
                    'box': [x1, y1, x2, y2],
                    'conf': float(conf)
                })
        
        self.update_tracks(detections)
        abandoned_bags = self.detect_abandoned_bags()
        annotated_frame = self._annotate_frame(frame, abandoned_bags)
        
        # Update FPS
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.detector_fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        return annotated_frame, abandoned_bags

    def _annotate_frame(self, frame, abandoned_bags):
        h, w = frame.shape[:2]
        
        # Draw persons
        for pid, person_data in self.person_tracker.items():
            x1, y1, x2, y2 = person_data['box']
            color = self.class_colors[person_data['class_id']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"P{pid}"
            if person_data['class_id'] == 3:
                label += " (with bag)"
            
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw bags
        for bid, bag_data in self.bag_tracker.items():
            x1, y1, x2, y2 = bag_data['box']
            bag_still_frames = self.bag_is_still.get(bid, 0)
            owner_id = self.bag_owners.get(bid)
            
            if bid in self.bag_colors:
                bag_color = self.bag_colors[bid]
            else:
                bag_color = self.class_colors[bag_data['class_id']]
            
            if bid in self.bag_alerted:
                # ABANDONED - RED ALERT
                status_color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 3)
                
                label = f"ABANDONED {bag_data['class_name'].upper()}"
                if owner_id is not None:
                    label += f" (Owner: P{owner_id})"
                
                cv2.putText(frame, label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            elif bid in self.bag_abandon_start:
                # COUNTDOWN - ORANGE WARNING
                status_color = (0, 165, 255)
                time_left = self.alert_time - (time.time() - self.bag_abandon_start[bid])
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                
                label = f"{bag_data['class_name']} (owner left) - {time_left:.1f}s"
                cv2.putText(frame, label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            elif bag_still_frames > 0:
                # STILL MONITORING - YELLOW/CYAN
                status_color = (255, 255, 0)
                progress = min(100, int(bag_still_frames * 100 / self.stillness_frames))
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                
                if owner_id is not None:
                    label = f"{bag_data['class_name']} (P{owner_id}) - {progress}% still"
                else:
                    label = f"{bag_data['class_name']} - {progress}% still"
                
                cv2.putText(frame, label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            else:
                # MOVING - NO ALERT
                cv2.rectangle(frame, (x1, y1), (x2, y2), bag_color, 2)
                
                if owner_id is not None:
                    label = f"{bag_data['class_name']} B{bid} (Owner: P{owner_id})"
                else:
                    label = f"{bag_data['class_name']} B{bid}"
                
                cv2.putText(frame, label, (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bag_color, 1)
            
            # Draw line from bag to owner
            if owner_id is not None:
                if owner_id in self.person_tracker:
                    # Owner visible - draw line
                    owner_center = self.person_tracker[owner_id]['center']
                    cv2.line(frame, bag_data['center'], owner_center, bag_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, bag_data['center'], 5, bag_color, -1)
                    cv2.circle(frame, owner_center, 5, bag_color, -1)
                else:
                    # Owner not visible (left camera) - still draw indicator
                    cv2.circle(frame, bag_data['center'], 8, (0, 0, 255), 2)
                    cv2.putText(frame, "OWNER LEFT", (bag_data['center'][0]-30, bag_data['center'][1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw alerts at top
        for i, alert in enumerate(abandoned_bags):
            text = f"ALERT: Abandoned {alert['class_name'].upper()} - {alert['duration']:.1f}s"
            cv2.putText(frame, text, (10, 40 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw info
        cv2.putText(frame, f'FPS: {self.detector_fps}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'People: {len(self.person_tracker)} | Bags: {len(self.bag_tracker)}', 
                   (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


def main():
    parser = argparse.ArgumentParser(description='4-Class Unattended Baggage Detection')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Model: yolov8n.pt (nano-fastest) or best.pt (custom)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0=webcam, or path to video)')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='Confidence threshold')
    parser.add_argument('--distance', type=int, default=150,
                       help='Max distance between person and bag (pixels)')
    parser.add_argument('--alert-time', type=int, default=5,
                       help='Seconds before alert (after person leaves)')
    parser.add_argument('--stillness-frames', type=int, default=30,
                       help='Frames bag must be still before monitoring')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (skip=2 means process every 2nd frame)')
    parser.add_argument('--resize', type=int, default=640,
                       help='Input image size (smaller = faster, 416/512/640)')
    
    args = parser.parse_args()
    
    try:
        detector = UnattendedBagDetector(
            model_path=args.model,
            max_owner_distance=args.distance,
            alert_time=args.alert_time,
            conf_threshold=args.conf,
            stillness_frames=args.stillness_frames
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"\nOpening video source: {args.source}")
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"\nResolution: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Skip frames: {args.skip_frames} (process every {args.skip_frames} frame)")
    print(f"Resize to: {args.resize}x{args.resize}")
    
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_video, fourcc, fps, (frame_width, frame_height))
    
    print("\nStarting detection... Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store original for drawing
            original_frame = frame.copy()
            
            # Process frame
            annotated_frame, abandoned_bags = detector.process_frame(frame, skip_frames=args.skip_frames, resize=args.resize)
            
            # Resize back to original if needed
            if annotated_frame.shape != original_frame.shape:
                annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
            
            if out:
                out.write(annotated_frame)
            
            cv2.imshow('Unattended Baggage Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


if __name__ == '__main__':
    main()