"""Hand detection using the MediaPipe Tasks API (works with current MediaPipe versions)."""
import os, urllib.request, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"

def get_detector(num_hands=2, conf=0.3):
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
    return vision.HandLandmarker.create_from_options(opts)

def detect(detector, rgb):
    """rgb: uint8 array (H,W,3). Returns list of hands; each hand = list of 21 landmarks (.x .y .z)."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return detector.detect(img).hand_landmarks


from types import SimpleNamespace as _NS

def detect_robust(detector, rgb):
    """Try the plain image, then a mirrored copy, then zoomed windows (for small hands)."""
    h, w = rgb.shape[:2]
    res = detect(detector, rgb)
    if res:
        return res
    res = detect(detector, rgb[:, ::-1])                      # mirrored; map x back
    if res:
        return [[_NS(x=1 - p.x, y=p.y, z=p.z) for p in H] for H in res]
    for x0, y0, x1, y1 in [(0.15, 0.15, 0.85, 0.85), (0, 0, 0.7, 0.7), (0.3, 0, 1, 0.7),
                           (0, 0.3, 0.7, 1), (0.3, 0.3, 1, 1), (0, 0.15, 1, 0.85)]:
        crop = rgb[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        res = detect(detector, crop)
        if res:                                               # map window coords back
            return [[_NS(x=x0 + p.x * (x1 - x0), y=y0 + p.y * (y1 - y0), z=p.z) for p in H] for H in res]
    return []
