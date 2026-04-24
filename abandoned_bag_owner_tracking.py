import cv2
import argparse
import time
import math
from collections import deque
from ultralytics import YOLO


# ─────────────────────────── helpers ───────────────────────────

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    xA, yA = max(ax1, bx1), max(ay1, by1)
    xB, yB = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    areaB = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(areaA + areaB - inter)


def scale_box(box, scale):
    """Scale a box that was detected on a resized frame back to original frame size."""
    x1, y1, x2, y2 = box
    return [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]


# ─────────────────────────── person ───────────────────────────

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


# ─────────────────────────── bag ───────────────────────────

class Bag:
    def __init__(self, bid, box):
        self.id = bid
        self.box = box
        self.center = center(box)
        self.last_seen = time.time()
        self.missed_frames = 0

        # position history for static check
        self.history = deque(maxlen=30)
        self.history.append(self.center)

        # smoothed center to reduce jitter
        self.smooth_center = self.center

        # owner tracking
        self.owner_id = None
        self.owner_last_center = None
        self.owner_last_seen = None
        self.near_owner_frames = 0

        # timers
        self.static_since = None
        self.alone_since = None
        self.abandoned_since = None
        self.is_abandoned = False

    def update(self, box):
        self.box = box
        new_center = center(box)

        # smooth center using EMA to reduce jitter-based false static resets
        alpha = 0.4
        self.smooth_center = (
            int(alpha * new_center[0] + (1 - alpha) * self.smooth_center[0]),
            int(alpha * new_center[1] + (1 - alpha) * self.smooth_center[1]),
        )
        self.center = new_center
        self.last_seen = time.time()
        self.missed_frames = 0
        self.history.append(self.smooth_center)

    def is_static(self, threshold=18):
        """Check if bag has been mostly stationary using smoothed center history."""
        if len(self.history) < 8:
            return False
        avg_x = sum(p[0] for p in self.history) / len(self.history)
        avg_y = sum(p[1] for p in self.history) / len(self.history)
        return dist(self.smooth_center, (avg_x, avg_y)) < threshold


# ─────────────────────────── person tracker ───────────────────────────

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
            best_i, best_score = None, -1e9
            for i, box in enumerate(person_boxes):
                if i in used:
                    continue
                c = center(box)
                d = dist(old.center, c)
                ov = iou(old.box, box)
                if not (ov > 0.3 or d < 60):
                    continue
                score = ov * 1000 - d
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                old.update(person_boxes[best_i])
                updated[pid] = old
                used.add(best_i)
            elif now - old.last_seen <= self.forget_sec:
                updated[pid] = old

        for i, box in enumerate(person_boxes):
            if i in used:
                continue
            c = center(box)
            if not any(dist(t.center, c) < 50 for t in updated.values()):
                updated[self.next_id] = Person(self.next_id, box)
                self.next_id += 1

        self.tracks = updated


# ─────────────────────────── bag tracker ───────────────────────────

class BagTracker:
    def __init__(self, max_match_dist=150, max_missed_frames=15):
        self.max_match_dist = max_match_dist
        self.max_missed_frames = max_missed_frames
        self.tracks = {}
        self.next_id = 0

    def update(self, bag_boxes):
        now = time.time()
        updated = {}
        used = set()

        for bid, old in self.tracks.items():
            best_i, best_score = None, -1e9
            for i, box in enumerate(bag_boxes):
                if i in used:
                    continue
                c = center(box)
                d = dist(old.smooth_center, c)
                ov = iou(old.box, box)
                score = ov * 1000 - d
                if (ov > 0.2 or d < 80) and score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                old.update(bag_boxes[best_i])
                updated[bid] = old
                used.add(best_i)
            else:
                old.missed_frames += 1
                # keep abandoned bags alive much longer even if not detected
                if old.is_abandoned:
                    if (now - old.last_seen) <= 15.0:
                        updated[bid] = old
                else:
                    if old.missed_frames <= self.max_missed_frames and (now - old.last_seen) <= 5.0:
                        updated[bid] = old

        for i, box in enumerate(bag_boxes):
            if i in used:
                continue
            c = center(box)
            if not any(dist(t.smooth_center, c) < 55 for t in updated.values()):
                updated[self.next_id] = Bag(self.next_id, box)
                self.next_id += 1

        self.tracks = updated


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser("Abandoned bag detection")
    parser.add_argument("--model",            type=str,   default="best.pt")
    parser.add_argument("--source",           type=str,   default="0")
    parser.add_argument("--conf",             type=float, default=0.40)   # lowered for stationary bags
    parser.add_argument("--near-person-dist", type=float, default=120.0)
    parser.add_argument("--bag-static-sec",   type=float, default=3.0)
    parser.add_argument("--bag-alone-sec",    type=float, default=4.0)
    parser.add_argument("--resize",           type=int,   default=640)
    parser.add_argument("--skip-frames",      type=int,   default=1)
    args = parser.parse_args()

    model = YOLO(args.model)

    # open source
    if args.source.startswith("rtsp://"):
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(0 if args.source == "0" else args.source)

    if not cap.isOpened():
        print("ERROR: could not open source.")
        return

    print("Classes:", model.names)

    # identify class IDs
    bag_ids, person_ids = set(), set()
    for cid, name in model.names.items():
        n = str(name).lower()
        if "bag" in n or "luggage" in n or "suitcase" in n or "backpack" in n:
            bag_ids.add(cid)
        if "person" in n:
            person_ids.add(cid)

    if not bag_ids or not person_ids:
        print("ERROR: model must have both 'bag' and 'person' classes.")
        print("Detected classes:", model.names)
        return

    person_tracker = PersonTracker(max_match_dist=90, forget_sec=5.0)
    bag_tracker    = BagTracker(max_match_dist=130, max_missed_frames=15)

    fps, fps_cnt = 0.0, 0
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

        # resize for inference, keep scale for drawing on original frame
        if w > args.resize:
            scale = args.resize / w
            infer_frame = cv2.resize(frame, (args.resize, int(h * scale)))
        else:
            scale = 1.0
            infer_frame = frame.copy()

        res = model(infer_frame, conf=args.conf, verbose=False)[0]

        person_boxes = []
        bag_boxes    = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box, conf_val, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cid = int(cls)

                # scale back to original frame size
                sx1, sy1, sx2, sy2 = scale_box([x1, y1, x2, y2], scale)

                if cid in person_ids:
                    person_boxes.append([sx1, sy1, sx2, sy2])
                elif cid in bag_ids:
                    bw, bh = sx2 - sx1, sy2 - sy1
                    if bw > 25 and bh > 25:
                        bag_boxes.append([sx1, sy1, sx2, sy2])

        person_tracker.update(person_boxes)
        bag_tracker.update(bag_boxes)

        now = time.time()

        # ────────── abandoned logic ──────────
        for bid, b in bag_tracker.tracks.items():

            # find nearest person
            nearest_pid, nearest_d = None, float("inf")
            for pid, p in person_tracker.tracks.items():
                d = dist(b.smooth_center, p.center)
                if d < nearest_d:
                    nearest_d = d
                    nearest_pid = pid

            person_close = nearest_pid is not None and nearest_d <= args.near_person_dist

            # ── owner assignment ──
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
                # update owner position if still visible
                if b.owner_id in person_tracker.tracks:
                    b.owner_last_center = person_tracker.tracks[b.owner_id].center
                    b.owner_last_seen = now

                # reassign owner only if bag is not abandoned and a much closer person appears
                if (not b.is_abandoned
                        and person_close
                        and nearest_pid != b.owner_id
                        and nearest_d < args.near_person_dist * 0.5):
                    b.owner_id = nearest_pid
                    b.owner_last_center = person_tracker.tracks[nearest_pid].center
                    b.owner_last_seen = now

            # ── static timer ──
            # FIX: only clear abandoned state if bag physically moves, not on tiny jitter
            if b.is_static():
                if b.static_since is None:
                    b.static_since = now
            else:
                # bag is moving — only reset if not abandoned
                if not b.is_abandoned:
                    b.static_since = None
                    b.alone_since  = None

            # ── alone timer ──
            if b.owner_id is not None:
                owner_visible = b.owner_id in person_tracker.tracks

                if owner_visible:
                    owner_center = person_tracker.tracks[b.owner_id].center
                    b.owner_last_center = owner_center
                    b.owner_last_seen = now
                    d_owner = dist(b.smooth_center, owner_center)

                    if d_owner > args.near_person_dist:
                        if b.alone_since is None:
                            b.alone_since = now
                    else:
                        # person is back near bag — reset abandoned state
                        b.alone_since     = None
                        b.is_abandoned    = False
                        b.abandoned_since = None
                else:
                    # owner disappeared from frame
                    if b.owner_last_seen is not None and (now - b.owner_last_seen) > 2.0:
                        if b.alone_since is None:
                            b.alone_since = now

            # ── abandoned condition ──
            if (
                b.owner_id is not None
                and b.static_since is not None
                and b.alone_since is not None
                and (now - b.static_since) >= args.bag_static_sec
                and (now - b.alone_since)  >= args.bag_alone_sec
            ):
                if not b.is_abandoned:
                    b.is_abandoned    = True
                    b.abandoned_since = now

                # keep alive even if detector misses it
                b.last_seen    = now
                b.missed_frames = 0

            # ── recovery: any person comes close to abandoned bag ──
            if b.is_abandoned and person_close:
                b.is_abandoned    = False
                b.abandoned_since = None
                b.alone_since     = None
                b.static_since    = None
                # reassign owner to whoever came back
                b.owner_id         = nearest_pid
                b.owner_last_center = person_tracker.tracks[nearest_pid].center
                b.owner_last_seen   = now

        # ────────── drawing on ORIGINAL frame ──────────

        # draw persons
        for pid, p in person_tracker.tracks.items():
            x1, y1, x2, y2 = p.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"P{pid}",
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        alerts = []

        for bid, b in bag_tracker.tracks.items():
            x1, y1, x2, y2 = b.box

            # determine state and color
            if b.is_abandoned:
                color = (0, 0, 255)           # red
                dur   = now - b.abandoned_since if b.abandoned_since else 0
                label = f"ABANDONED B{bid} ({dur:.1f}s)"
                alerts.append(label)

            elif (b.owner_id is not None
                  and b.static_since is not None
                  and b.alone_since is not None):
                # warning phase — bag is static and alone but timer not expired yet
                color = (0, 165, 255)         # orange
                alone_t  = now - b.alone_since
                remain   = max(0, args.bag_alone_sec - alone_t)
                label    = f"WARNING B{bid} ({remain:.1f}s)"

            elif b.owner_id is not None:
                color = (255, 180, 0)         # blue — owner assigned, bag active
                label = f"BAG B{bid}"

            else:
                color = (200, 200, 200)       # grey — no owner yet
                label = f"BAG B{bid}"

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, b.smooth_center, 4, color, -1)

            # draw owner line ONLY during warning phase and active (not abandoned)
            if not b.is_abandoned and b.owner_id is not None:
                if b.owner_id in person_tracker.tracks:
                    owner = person_tracker.tracks[b.owner_id]

                    # line color: orange during warning, white when normal
                    line_color = (0, 165, 255) if b.alone_since is not None else (255, 255, 255)

                    cv2.line(frame, b.smooth_center, owner.center,
                             line_color, 2, cv2.LINE_AA)

                    cv2.putText(frame, f"Owner P{b.owner_id}",
                                (b.smooth_center[0] + 6, b.smooth_center[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)

        # draw alerts top-left
        for i, txt in enumerate(alerts):
            cv2.putText(frame, txt,
                        (10, 45 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # FPS counter
        fps_cnt += 1
        if now - fps_t0 >= 1.0:
            fps     = fps_cnt / (now - fps_t0)
            fps_cnt = 0
            fps_t0  = now

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Abandoned Bag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()