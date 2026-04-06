# live_camera.py
"""
Live webcam + YOLOv8 inference with:
 - face-check for hair cap -> generates `no_hair_cap`
 - detection of critical items -> raises alert if seen a few frames in a row
 - when alert active: visual banner + optional beep + temporary sensitivity adjustment
"""

import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import threading

# ========== CONFIG ==========
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
IMG_WIDTH = 1280
IMG_HEIGHT = 720
BASE_CONFIDENCE = 0.35   # base detection confidence
USE_HALF = True          # use fp16 on GPU
MIN_IOU_FOR_CAP = 0.30
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 8  # Increased to prevent false positive faces on walls

# ALERT settings
CRITICAL_LABELS = {"rat", "lizard", "cockroach", "no_gloves", "no_hair_cap"}
ALERT_FRAMES_NEEDED = 3        # number of consecutive frames with at least one critical detection to trigger
ALERT_HOLD_SECS = 6            # how long alert stays active (seconds)
ALERT_CONF_ADJUST = -0.15      # temporary adjust to confidence during alert (negative => more sensitive)
BEEP_ON_ALERT = True           # attempt a beep on alert (Windows winsound used if available)
# ============================

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {DEVICE}")

# Load model
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)

# find hair_cap index helper
def find_class_index(names, target_variants=("hair_cap", "hair cap", "cap", "haircap", "hair-cap")):
    if names is None:
        return None
    for i, name in enumerate(names):
        n = str(name).lower().strip()
        if n in target_variants or n.replace(" ", "_") in target_variants:
            return i
    # fallback heuristics
    for i, name in enumerate(names):
        n = str(name).lower()
        if "hair" in n and "cap" in n:
            return i
    return None

hair_cap_index = find_class_index(getattr(model, "names", None))
if hair_cap_index is None:
    print("[WARN] 'hair_cap' not found in model.names. fuzzy matching will be used for checking 'cap' words.")
else:
    print(f"[INFO] hair_cap index: {hair_cap_index} (name='{model.names[hair_cap_index]}')")

# Haar face detector
haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
if not os.path.exists(haar_path):
    raise FileNotFoundError("Haar cascade not found at: " + haar_path)
face_cascade = cv2.CascadeClassifier(haar_path)

def xywh_to_xyxy(x, y, w, h):
    return (int(x), int(y), int(x + w), int(y + h))

def iou_xyxy(boxA, boxB):
    # boxA is the face, boxB is the cap
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    # Use intersection over the face area, not the union!
    return interArea / boxAArea if boxAArea > 0 else 0.0

def draw_detection(frame, xyxy, label, conf, color=(8,163,8)):
    x1, y1, x2, y2 = map(int, xyxy)
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

def draw_face_no_cap(frame, xyxy_face):
    x1, y1, x2, y2 = map(int, xyxy_face)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text = "no_hair_cap"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0,0,255), -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# beep helper (non-blocking)
def beep_once():
    try:
        # Windows
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        # fallback to terminal bell (may not sound)
        print("\a", end="", flush=True)

def alert_beep_thread():
    if BEEP_ON_ALERT:
        t = threading.Thread(target=beep_once, daemon=True)
        t.start()

def draw_alert_banner(frame, texts):
    h, w = frame.shape[:2]
    banner_h = 80
    cv2.rectangle(frame, (0, 0), (w, banner_h), (0, 0, 180), -1)
    txt = "ALERT: " + ", ".join(texts)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, txt, (12, banner_h//2 + th//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")

    fps_deque = deque(maxlen=12)
    last_print = time.time()

    # alert state
    alert_frame_counter = 0
    alert_active_until = 0.0
    alert_detected_items = set()

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

            # face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=FACE_SCALE_FACTOR, 
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=(100, 100)
            )
            faces_xyxy = [xywh_to_xyxy(x, y, w, h) for (x, y, w, h) in faces]

            # choose confidence (base or temporarily adjusted during alert)
            now = time.time()
            alert_now = now < alert_active_until
            conf = BASE_CONFIDENCE
            if alert_now:
                # ALERT_CONF_ADJUST negative makes it more sensitive; positive would make stricter.
                conf = max(0.01, min(0.99, BASE_CONFIDENCE + ALERT_CONF_ADJUST))

            half_flag = True if (DEVICE != "cpu" and USE_HALF) else False
            results = model.predict(source=frame, device=DEVICE, conf=conf, half=half_flag, verbose=False)

            det_boxes = []
            det_conf = []
            det_cls = []
            det_label = []

            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                try:
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback
                    xyxy_arr = []
                    confs = []
                    clss = []
                    for b in boxes:
                        try:
                            xyxy_arr.append(b.xyxy[0].tolist())
                            confs.append(float(b.conf[0]))
                            clss.append(int(b.cls[0]))
                        except Exception:
                            pass
                for xb, cf, cc in zip(xyxy_arr, confs, clss):
                    det_boxes.append(tuple(map(float, xb)))
                    det_conf.append(float(cf))
                    det_cls.append(int(cc))
                    lbl = model.names[int(cc)] if hasattr(model, "names") else str(int(cc))
                    det_label.append(lbl)

            # Determine faces without hair_cap and add synthetic 'no_hair_cap' detections
            synthetic_no_hair = []
            for face_xy in faces_xyxy:
                has_cap = False
                if hair_cap_index is not None:
                    for xb, cc in zip(det_boxes, det_cls):
                        if cc == hair_cap_index and iou_xyxy(face_xy, xb) >= MIN_IOU_FOR_CAP:
                            has_cap = True
                            break
                else:
                    for xb, lbl in zip(det_boxes, det_label):
                        lower = lbl.lower()
                        if ("cap" in lower) or ("hair" in lower):
                            if iou_xyxy(face_xy, xb) >= MIN_IOU_FOR_CAP:
                                has_cap = True
                                break
                if not has_cap:
                    # synthetic label placed at face bbox center
                    synthetic_no_hair.append(face_xy)
                    # add to lists so it will be considered for alerts below & drawn
                    det_boxes.append(tuple(map(float, face_xy)))
                    det_conf.append(1.0)
                    det_cls.append(-1)  # synthetic index
                    det_label.append("no_hair_cap")

            # Draw all model detections (green for good, red for bad)
            for xb, cf, cc, lbl in zip(det_boxes, det_conf, det_cls, det_label):
                if cc == -1 and lbl == "no_hair_cap":
                    # skip here: draw synthetic Haar faces in special style below
                    continue
                
                if "no_" in lbl.lower() or lbl.lower() in CRITICAL_LABELS:
                    draw_detection(frame, xb, lbl, cf, color=(0, 0, 255)) # Red for violations
                else:
                    draw_detection(frame, xb, lbl, cf, color=(8,163,8)) # Green for ok

            # Draw synthetic no_hair_cap (red)
            for face_xy in synthetic_no_hair:
                draw_face_no_cap(frame, face_xy)

            # Check for critical items in this frame (case-insensitive)
            detected_critical = set()
            for lbl in det_label:
                l = lbl.lower()
                for crit in CRITICAL_LABELS:
                    if crit in l:   # substring match
                        detected_critical.add(crit)

            # Update alert counter
            if detected_critical:
                alert_frame_counter += 1
                # record items
                alert_detected_items.update(detected_critical)
            else:
                # decay counter slowly if no critical seen
                alert_frame_counter = max(0, alert_frame_counter - 1)

            # If enough consecutive frames contain critical detections and alert not active: trigger
            if alert_frame_counter >= ALERT_FRAMES_NEEDED and time.time() >= alert_active_until:
                alert_active_until = time.time() + ALERT_HOLD_SECS
                # On alert, create banner & beep
                alert_beep_thread()
                print(f"[ALERT] detected: {sorted(alert_detected_items)}")
                # (we intentionally do not reset alert_detected_items here to show all items seen)

            # If alert is active, draw banner with items
            if time.time() < alert_active_until:
                # show banner with items seen
                texts = sorted(list(alert_detected_items)) if alert_detected_items else ["unknown"]
                draw_alert_banner(frame, texts)
            else:
                # when alert period finished, clear recorded items (so next alerts are fresh)
                if alert_detected_items:
                    # keep a short time to avoid flicker; only clear when fully expired
                    alert_detected_items.clear()

            # Overlay FPS and device
            frame_end = time.time()
            fps = 1.0 / (frame_end - frame_start) if (frame_end - frame_start) > 0 else 0.0
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220,220,220), 2)
            device_text = f"DEVICE: {'CUDA' if DEVICE != 'cpu' else 'CPU'}"
            cv2.putText(frame, device_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("YOLOv8 Live (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time.time() - last_print > 5:
                print(f"[INFO] avg FPS: {avg_fps:.2f} | conf={conf:.2f} | faces={len(faces_xyxy)} | dets={len(det_boxes)} | alert_count={alert_frame_counter}")
                last_print = time.time()

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
