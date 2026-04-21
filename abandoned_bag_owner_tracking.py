import cv2
import argparse
import time
import math
from collections import deque
from ultralytics import YOLO


# ---------------- helpers ----------------

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    areaB = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(areaA + areaB - inter)


# ---------------- person tracking ----------------

class Person:
    def __init__(self, pid, box):
        self.id = pid
        self.box = box
        self.center = center(box)
        self.last_seen = time.time()

    def update(self, box):
        self.box = box
        self.center = center(box)
        self.last_seen = time.time()


# ---------------- bag tracking ----------------

class Bag:
    def __init__(self, bid, box):
        self.id = bid
        self.box = box
        self.center = center(box)
        self.last_seen = time.time()
        self.missed_frames = 0

        self.history = deque(maxlen=20)
        self.history.append(self.center)

        self.owner_id = None
        self.owner_last_center = None
        self.owner_last_seen = None

        self.near_owner_frames = 0

        self.static_since = None
        self.alone_since = None
        self.abandoned_since = None
        self.is_abandoned = False

    def update(self, box):
        self.box = box
        self.center = center(box)
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history.append(self.center)

    def is_static(self, threshold=15):
        if len(self.history) < 5:
            return False
        avg_x = sum(p[0] for p in self.history) / len(self.history)
        avg_y = sum(p[1] for p in self.history) / len(self.history)
        return dist(self.center, (avg_x, avg_y)) < threshold


# ---------------- trackers ----------------

class PersonTracker:
    def __init__(self, max_match_dist=90, forget_sec=5.0):
        self.max_match_dist = max_match_dist
        self.forget_sec = forget_sec
        self.tracks = {}
        self.next_id = 0

    def update(self, person_boxes):
        now = time.time()
        updated = {}
        used = set()

        for pid, old in self.tracks.items():
            best_i = None
            best_score = -1e9

            for i, box in enumerate(person_boxes):
                if i in used:
                    continue

                c = center(box)
                d = dist(old.center, c)
                ov = iou(old.box, box)

                # stricter matching to reduce ID jumps
                if not (ov > 0.3 or d < 50):
                    continue

                score = ov * 1000 - d

                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                old.update(person_boxes[best_i])
                updated[pid] = old
                used.add(best_i)
            else:
                if now - old.last_seen <= self.forget_sec:
                    updated[pid] = old

        for i, box in enumerate(person_boxes):
            if i in used:
                continue

            c = center(box)
            too_close = False
            for t in updated.values():
                if dist(t.center, c) < 50:
                    too_close = True
                    break

            if not too_close:
                updated[self.next_id] = Person(self.next_id, box)
                self.next_id += 1

        self.tracks = updated


class BagTracker:
    def __init__(self, max_match_dist=150, max_missed_frames=2, forget_abandoned_sec=10):
        self.max_match_dist = max_match_dist
        self.max_missed_frames = max_missed_frames
        self.forget_abandoned_sec = forget_abandoned_sec
        self.tracks = {}
        self.next_id = 0

    def update(self, bag_boxes):
        now = time.time()
        updated = {}
        used = set()

        for bid, old in self.tracks.items():
            best_i = None
            best_score = -1e9

            for i, box in enumerate(bag_boxes):
                if i in used:
                    continue
                c = center(box)
                d = dist(old.center, c)
                ov = iou(old.box, box)
                score = ov * 1000 - d

                if (ov > 0.25 or d < 65) and score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                old.update(bag_boxes[best_i])
                updated[bid] = old
                used.add(best_i)
            else:
                old.missed_frames += 1

                if old.is_abandoned:
                    updated[bid] = old
                else:
                    if old.missed_frames <= 15 and (now - old.last_seen) <= 5.0:
                        updated[bid] = old

        for i, box in enumerate(bag_boxes):
            if i in used:
                continue

            c = center(box)

            #prevent duplicate bag IDs
            too_close = False
            for t in updated.values():
                if dist(t.center, c) < 50:
                    too_close = True
                    break

            if not too_close:
                updated[self.next_id] = Bag(self.next_id, box)
                self.next_id += 1

        self.tracks = updated


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser("Abandoned bag detection")
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--near-person-dist", type=float, default=100.0)
    parser.add_argument("--bag-static-sec", type=float, default=3.0)
    parser.add_argument("--bag-alone-sec", type=float, default=3.0)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--skip-frames", type=int, default=1)
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        print("ERROR: could not open source.")
        return

    print("Classes:", model.names)

    bag_ids, person_ids = set(), set()
    for cid, name in model.names.items():
        n = str(name).lower()
        if "bag" in n:
            bag_ids.add(cid)
        if "person" in n:
            person_ids.add(cid)

    if not bag_ids or not person_ids:
        print("ERROR: model must have both 'bag' and 'person' classes.")
        return

    person_tracker = PersonTracker(max_match_dist=90, forget_sec=5.0)
    bag_tracker = BagTracker(max_match_dist=120, max_missed_frames=2, forget_abandoned_sec=10)

    fps = 0.0
    fps_cnt = 0
    fps_t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % args.skip_frames != 0:
            cv2.imshow("Abandoned Bag Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        h, w = frame.shape[:2]
        if w > args.resize:
            scale = args.resize / w
            infer_w = args.resize
            infer_h = int(h * scale)
            infer_frame = cv2.resize(frame, (infer_w, infer_h))
        else:
            infer_frame = frame.copy()

        res = model(infer_frame, conf=args.conf, verbose=False)[0]

        person_boxes = []
        bag_boxes = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cid = int(cls)

                if cid in person_ids:
                    person_boxes.append([x1, y1, x2, y2])

                elif cid in bag_ids:
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw > 30 and bh > 30:
                        bag_boxes.append([x1, y1, x2, y2])

        person_tracker.update(person_boxes)
        bag_tracker.update(bag_boxes)

        now = time.time()

        # -------- abandoned logic --------
        for bid, b in bag_tracker.tracks.items():
            nearest_pid = None
            nearest_d = float("inf")
            for pid, p in person_tracker.tracks.items():
                d = dist(b.center, p.center)
                if d < nearest_d:
                    nearest_d = d
                    nearest_pid = pid

            person_close = nearest_pid is not None and nearest_d <= args.near_person_dist

            # assign owner
            if b.owner_id is None:
                if person_close:
                    b.near_owner_frames += 1
                    if b.near_owner_frames >= 3:
                        b.owner_id = nearest_pid
                        b.owner_last_center = person_tracker.tracks[nearest_pid].center
                        b.owner_last_seen = now
                else:
                    b.near_owner_frames = 0
            else:
                if b.owner_id in person_tracker.tracks:
                    b.owner_last_center = person_tracker.tracks[b.owner_id].center
                    b.owner_last_seen = now

                if person_close and nearest_pid != b.owner_id and not b.is_static():
                    if nearest_d < args.near_person_dist * 0.55:
                        b.owner_id = nearest_pid
                        b.owner_last_center = person_tracker.tracks[nearest_pid].center
                        b.owner_last_seen = now

            # static timer
            if b.is_static():
                if b.static_since is None:
                    b.static_since = now
            else:
                b.static_since = None
                b.alone_since = None
                b.is_abandoned = False
                b.abandoned_since = None

            # alone timer
            if b.owner_id is not None:
                owner_visible = b.owner_id in person_tracker.tracks

                if owner_visible:
                    owner_center = person_tracker.tracks[b.owner_id].center
                    b.owner_last_center = owner_center
                    b.owner_last_seen = now
                    d_owner = dist(b.center, owner_center)

                    if d_owner > args.near_person_dist:
                        if b.alone_since is None:
                            b.alone_since = now
                    else:
                        b.alone_since = None
                else:
                    if b.owner_last_seen is not None and (now - b.owner_last_seen) > 2.0:
                        if b.alone_since is None:
                            b.alone_since = now

            # abandoned condition
            if (
                b.owner_id is not None
                and b.static_since is not None
                and b.alone_since is not None
                and (now - b.static_since) >= args.bag_static_sec
                and (now - b.alone_since) >= args.bag_alone_sec
            ):
                if not b.is_abandoned:
                    b.is_abandoned = True
                    b.abandoned_since = now

                b.last_seen = now
                b.missed_frames = 0

        # -------- drawing --------

        for pid, p in person_tracker.tracks.items():
            x1, y1, x2, y2 = p.box
            cv2.rectangle(infer_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                infer_frame, f"P{pid}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        alerts = []

        for bid, b in bag_tracker.tracks.items():
            x1, y1, x2, y2 = b.box

            if b.is_abandoned:
                color = (0, 0, 255)
                label = f"ABANDONED BAG B{bid}"
                if b.abandoned_since is not None:
                    label += f" ({now - b.abandoned_since:.1f}s)"
                alerts.append(label)
            elif b.owner_id is not None and b.static_since is not None and b.alone_since is not None:
                color = (0, 165, 255)
                label = f"B{bid} static {now - b.static_since:.1f}s alone {now - b.alone_since:.1f}s"
            else:
                color = (255, 0, 0)
                label = f"BAG B{bid}"

            cv2.rectangle(infer_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                infer_frame, label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            cv2.circle(infer_frame, b.center, 4, color, -1)

            # draw line ONLY to assigned owner, and only while not abandoned
            if not b.is_abandoned and b.owner_id is not None:
                if b.owner_id in person_tracker.tracks:
                    owner = person_tracker.tracks[b.owner_id]

                    cv2.line(
                        infer_frame,
                        b.center,
                        owner.center,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        infer_frame,
                        f"Owner P{b.owner_id}",
                        (b.center[0] + 5, b.center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

        for i, txt in enumerate(alerts):
            cv2.putText(
                infer_frame, txt,
                (10, 40 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        fps_cnt += 1
        if now - fps_t0 >= 1.0:
            fps = fps_cnt / (now - fps_t0)
            fps_cnt = 0
            fps_t0 = now

        cv2.putText(
            infer_frame, f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Abandoned Bag Detection", infer_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()