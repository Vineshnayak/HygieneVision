import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class HygieneEngine:
    def __init__(self, model_path="best.pt", device=None):
        self.device = device if device else (0 if torch.cuda.is_available() else "cpu")
        print(f"[HygieneEngine] Loading YOLO model from {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        
        # Haar face detector
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(haar_path):
            print(f"[HygieneEngine WARN] Haar cascade not found at: {haar_path}")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(haar_path)

        # Configs
        self.face_scale_factor = 1.1
        self.face_min_neighbors = 8  # Increased to prevent false positive faces on walls
        self.min_iou_for_cap = 0.30
        self.critical_labels = {"rat", "lizard", "cockroach", "no_gloves", "no_hair_cap"}
        self.alert_frames_needed = 3
        self.alert_hold_secs = 6

        # State for alerts
        self.alert_frame_counter = 0
        self.alert_active_until = 0.0
        self.alert_detected_items = set()

        self.hair_cap_index = self._find_class_index(getattr(self.model, "names", None))

    def _find_class_index(self, names, target_variants=("hair_cap", "hair cap", "cap", "haircap", "hair-cap")):
        if names is None:
            return None
        for i, name in enumerate(names.values() if isinstance(names, dict) else names):
            n = str(name).lower().strip()
            if n in target_variants or n.replace(" ", "_") in target_variants:
                return i
        for i, name in enumerate(names.values() if isinstance(names, dict) else names):
            n = str(name).lower()
            if "hair" in n and "cap" in n:
                return i
        return None

    def _xywh_to_xyxy(self, x, y, w, h):
        return (int(x), int(y), int(x + w), int(y + h))

    def _overlap_metric(self, face_box, cap_box):
        xA = max(face_box[0], cap_box[0])
        yA = max(face_box[1], cap_box[1])
        xB = min(face_box[2], cap_box[2])
        yB = min(face_box[3], cap_box[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        faceArea = max(0, face_box[2] - face_box[0]) * max(0, face_box[3] - face_box[1])
        # Use intersection over the face area, not the union! 
        # This prevents large cap boxes from punishing the IoU score.
        return interArea / faceArea if faceArea > 0 else 0.0

    def draw_detection(self, frame, xyxy, label, conf, color=(8,163,8)):
        x1, y1, x2, y2 = map(int, xyxy)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    def draw_face_no_cap(self, frame, xyxy_face):
        x1, y1, x2, y2 = map(int, xyxy_face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "no_hair_cap"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0,0,255), -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    def process_frame(self, frame, conf_threshold=0.35, draw=True):
        """
        Runs YOLO inference alongside Haar cascade to detect hygiene issues.
        Modifies `frame` in place if draw=True.
        Returns: frame, active_alert_items (list of strings or None), detections_list
        """
        original_shape = frame.shape
        # Optional resizing can be done outside, but let's do predictions on the raw frame
        
        faces_xyxy = []
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Add minSize to ignore small background shapes like light switches
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=self.face_scale_factor, 
                minNeighbors=self.face_min_neighbors,
                minSize=(100, 100)
            )
            faces_xyxy = [self._xywh_to_xyxy(x, y, w, h) for (x, y, w, h) in faces]

        # Use half precision if on GPU
        use_half = True if str(self.device) != "cpu" else False
        results = self.model.predict(source=frame, device=self.device, conf=conf_threshold, half=use_half, verbose=False)

        det_boxes = []
        det_conf = []
        det_cls = []
        det_label = []

        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None: continue
            
            try:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # Fallback for some YOLO formats
                xyxy_arr = []
                confs = []
                clss = []
                for b in boxes:
                    xyxy_arr.append(b.xyxy[0].tolist())
                    confs.append(float(b.conf[0]))
                    clss.append(int(b.cls[0]))

            for xb, cf, cc in zip(xyxy_arr, confs, clss):
                det_boxes.append(tuple(map(float, xb)))
                det_conf.append(float(cf))
                det_cls.append(int(cc))
                
                names_dict = getattr(self.model, "names", {})
                lbl = names_dict.get(int(cc), str(cc)) if names_dict else str(cc)
                det_label.append(lbl)

        # Determine faces without hair_cap
        synthetic_no_hair = []
        for face_xy in faces_xyxy:
            has_cap = False
            if self.hair_cap_index is not None:
                for xb, cc in zip(det_boxes, det_cls):
                    if cc == self.hair_cap_index and self._overlap_metric(face_xy, xb) >= self.min_iou_for_cap:
                        has_cap = True
                        break
            else:
                for xb, lbl in zip(det_boxes, det_label):
                    lower = lbl.lower()
                    if ("cap" in lower) or ("hair" in lower):
                        if self._overlap_metric(face_xy, xb) >= self.min_iou_for_cap:
                            has_cap = True
                            break
            
            if not has_cap:
                synthetic_no_hair.append(face_xy)
                det_boxes.append(tuple(map(float, face_xy)))
                det_conf.append(1.0)
                det_cls.append(-1)
                det_label.append("no_hair_cap")

        if draw:
            for xb, cf, cc, lbl in zip(det_boxes, det_conf, det_cls, det_label):
                if cc == -1 and lbl == "no_hair_cap":
                    continue # Synthesized Haar faces are drawn specifically below
                
                if "no_" in lbl.lower() or lbl.lower() in self.critical_labels:
                    # Native violations (e.g., no_gloves, cockroach) drawn in RED
                    self.draw_detection(frame, xb, lbl, cf, color=(0, 0, 255))
                else:
                    self.draw_detection(frame, xb, lbl, cf, color=(34, 211, 238)) # Cyber blue

            for face_xy in synthetic_no_hair:
                self.draw_face_no_cap(frame, face_xy)

        # Critical item check
        detected_critical = set()
        for lbl in det_label:
            l = lbl.lower()
            for crit in self.critical_labels:
                if crit in l:
                    detected_critical.add(crit)

        # Real-time Status Update with subtle 3-frame debounce to prevent flashing
        if detected_critical:
            self.alert_frame_counter += 1
            if self.alert_frame_counter >= self.alert_frames_needed:
                self.alert_detected_items = detected_critical.copy()
        else:
            self.alert_frame_counter = max(0, self.alert_frame_counter - 1)
            # Instantly clear status if counter drops below threshold
            if self.alert_frame_counter < self.alert_frames_needed:
                self.alert_detected_items.clear()

        active_alerts = sorted(list(self.alert_detected_items)) if self.alert_detected_items else None
                
        # Return frame, alerts, and pure detections info for UI rendering
        detections_info = [
            {"label": lbl, "conf": cf, "bbox": xb}
            for lbl, cf, xb in zip(det_label, det_conf, det_boxes)
        ]
            
        return frame, active_alerts, detections_info
