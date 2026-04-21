# detection_module.py
"""
Lightweight wrapper around YOLOv8 for person + bag detection.

Usage:
    from detection_module import BagPersonDetector

    detector = BagPersonDetector(model_path="yolov8n.pt", device="cuda")
    detections = detector.detect(frame)
"""

import torch
from ultralytics import YOLO


class BagPersonDetector:
    """
    Runs YOLOv8 and returns a simple list of detections:
    [
        {
            "cls_id": int,
            "cls_name": str,
            "conf": float,
            "bbox": (x1, y1, x2, y2)
        },
        ...
    ]
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = None,
        conf_thres: float = 0.4,
    ):
        # Auto-select device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf_thres = conf_thres

        # Keep COCO class names for easy access
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Run YOLOv8 on a single BGR frame (numpy array from OpenCV).
        Returns a list of dicts: {cls_id, cls_name, conf, bbox}
        """
        results = self.model(frame, verbose=False)[0]

        detections = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

        return detections
