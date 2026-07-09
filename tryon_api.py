"""Virtual Try-On API.

Two backends, with graceful fallback:

1. MediaPipe Pose overlay (preferred when ``mediapipe`` is importable):
   detect shoulder / hip / foot landmarks → place, scale and rotate the cloth
   onto the torso by category, remove the product photo's background with
   HSV + GrabCut, and feather the alpha for a soft edge.

2. OpenCV fixed-ratio overlay (fallback when no human is detected in the
   photo, or when ``mediapipe`` is not installed): the previous behaviour —
   scale to ~45% of width and paste at chest level.

The full ACGPN pipeline is intentionally not used: this repo only has the
``label2city`` refine weights and is missing GMM / TOM / SCHP / OpenPose, and
ACGPN's old torch dependency does not install on Python 3.14.

Usage:
    python tryon_api.py            # serves on :8001

Endpoints:
    POST /tryon                       Generate try-on image
    GET  /tryon/categories            Available clothing categories
    GET  /tryon/clothing              Clothing items by category
    POST /tryon/upload/person         Upload person photo
    POST /tryon/upload/cloth         Upload cloth photo
"""
import math
import os
import re
import io
import base64
import uuid
import logging
import threading
from typing import Optional, List

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (paths anchored to this file so the service runs from any cwd)
_BASE_DIR = Path(__file__).resolve().parent
ACGPN_DIR = _BASE_DIR / "ACGPN"
STATIC_DIR = _BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

# Upload limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png"}

# Clothing categories that drive placement logic
VALID_CATEGORIES = {"top", "bottom", "dress", "outerwear", "shoes", "accessories"}

app = FastAPI(title="Virtual Try-On API", version="2.0.0")

# Auth uses a Bearer header (not cookies), so wildcard origins stay valid
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _validated_ext(filename: Optional[str], content_type: str) -> str:
    """Return a whitelisted image extension, or raise 400."""
    ext = ""
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        ext = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png"}.get(content_type, "")
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Only jpg/jpeg/png images are allowed")
    return ext


def _find_file(directory: Path, prefix: str, image_id: str) -> Optional[Path]:
    """Find an uploaded file by exact id, with NO substring/path-traversal match.

    ``image_id`` must be a safe token (alphanumeric, underscore, hyphen). The
    file is matched as ``{prefix}_{image_id}.{ext}`` via glob, anchored on the
    ``.`` after the id so it cannot match another id's file.
    """
    if not image_id or not re.match(r"^[A-Za-z0-9_-]+$", image_id):
        return None
    matches = list(directory.glob(f"{prefix}_{image_id}.*"))
    return matches[0] if matches else None


# ============ Data Models ============

class TryOnRequest(BaseModel):
    person_image_id: str
    cloth_image_id: str
    # Category drives where the cloth is placed on the body. Defaults to "top"
    # (the most common case and the previous behaviour).
    category: str = "top"


class ClothingItem(BaseModel):
    id: str
    name: str
    category: str
    image_url: str
    thumbnail_url: str


class TryOnResponse(BaseModel):
    result_id: str
    result_url: str
    explanation: str
    # Which backend actually produced the image (mediapipe / fallback)
    backend: str


# ============ Clothing Database (Mock) ============

CLOTHING_DATABASE = {
    "top": [
        {"id": "top_001", "name": "White Cotton T-Shirt", "category": "top", "image": "clothes/top_001.jpg"},
        {"id": "top_002", "name": "Blue Denim Shirt", "category": "top", "image": "clothes/top_002.jpg"},
        {"id": "top_003", "name": "Black Striped T-Shirt", "category": "top", "image": "clothes/top_003.jpg"},
        {"id": "top_004", "name": "Red Summer Dress", "category": "top", "image": "clothes/top_004.jpg"},
    ],
    "bottom": [
        {"id": "bottom_001", "name": "Blue Jeans", "category": "bottom", "image": "clothes/bottom_001.jpg"},
        {"id": "bottom_002", "name": "Black Shorts", "category": "bottom", "image": "clothes/bottom_002.jpg"},
        {"id": "bottom_003", "name": "Gray Trousers", "category": "bottom", "image": "clothes/bottom_003.jpg"},
    ],
    "dress": [
        {"id": "dress_001", "name": "Red Evening Gown", "category": "dress", "image": "clothes/dress_001.jpg"},
        {"id": "dress_002", "name": "Blue Summer Dress", "category": "dress", "image": "clothes/dress_002.jpg"},
    ],
    "outerwear": [
        {"id": "outer_001", "name": "Black Leather Jacket", "category": "outerwear", "image": "clothes/outer_001.jpg"},
        {"id": "outer_002", "name": "White Blazer", "category": "outerwear", "image": "clothes/outer_002.jpg"},
    ],
}


# ============ Backend detection ============

def check_tryon_backend() -> dict:
    """Report which try-on backends are importable."""
    status = {"opencv": False, "mediapipe": False}
    try:
        import cv2  # noqa: F401
        status["opencv"] = True
    except ImportError:
        pass
    try:
        import mediapipe  # noqa: F401
        status["mediapipe"] = True
    except ImportError:
        pass
    return status


def check_acgpn_installed() -> bool:
    """Back-compat shim — True when at least OpenCV is available."""
    return check_tryon_backend()["opencv"]


# ============ Pose detection (MediaPipe) ============

# MediaPipe Pose landmark indices (33 total). Only the ones we use.
_LANDMARK_L_SHOULDER = 11
_LANDMARK_R_SHOULDER = 12
_LANDMARK_L_HIP = 23
_LANDMARK_R_HIP = 24
_LANDMARK_L_FOOT = 31
_LANDMARK_R_FOOT = 32
_LANDMARK_NOSE = 0

# PoseLandmarker Tasks model (downloaded once; see TRYON_SETUP / README).
POSE_MODEL_PATH = STATIC_DIR / "models" / "pose_landmarker_lite.task"

# Module-level singleton: model load (~5 MB) takes a few hundred ms, so we
# reuse one detector across requests. Guarded by a lock for thread safety.
_pose_detector = None
_pose_detector_lock = threading.Lock()


def _get_pose_detector():
    """Lazy singleton for the MediaPipe PoseLandmarker.

    Uses the Tasks API (``mediapipe.tasks.python.vision``). The legacy
    ``mediapipe.solutions.pose`` is not shipped in the 0.10.35 win wheel for
    Python 3.14, so we go through Tasks instead. Raises if the model file is
    missing or mediapipe is not installed.
    """
    global _pose_detector
    if _pose_detector is not None:
        return _pose_detector
    with _pose_detector_lock:
        if _pose_detector is not None:
            return _pose_detector
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not POSE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pose model not found at {POSE_MODEL_PATH}. "
                "Download pose_landmarker_lite.task from "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        _pose_detector = PoseLandmarker.create_from_options(opts)
        return _pose_detector


def detect_pose_landmarks(person_bgr):
    """Run MediaPipe Pose once on the person image.

    Returns (pts, mirrored) where pts is a list of (x, y, visibility) tuples in
    pixel coords (33 entries), and mirrored is True when the person appears
    horizontally mirrored (selfie mode: left shoulder > right shoulder in image).
    """
    import cv2
    import mediapipe as mp

    h, w = person_bgr.shape[:2]
    rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector = _get_pose_detector()
    res = detector.detect(mp_img)
    if not res.pose_landmarks:
        return None, False
    lms = res.pose_landmarks[0]
    pts = [(lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)) for lm in lms]

    # Detect selfie mirroring: in a non-mirrored front-facing photo
    # the image-x of the left shoulder (11) < right shoulder (12).
    sh_l = pts[_LANDMARK_L_SHOULDER]
    sh_r = pts[_LANDMARK_R_SHOULDER]
    mirrored = sh_l[0] > sh_r[0]

    return pts, mirrored


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _best_ankle(pts):
    """Return the more-visible ankle landmark (left or right)."""
    a_l = pts[27]  # left ankle
    a_r = pts[28]  # right ankle
    return a_l if a_l[2] >= a_r[2] else a_r


def _knee_mid(pts):
    """Return midpoint between knees as a fallback hip proxy."""
    k_l = pts[25]
    k_r = pts[26]
    return ((k_l[0] + k_r[0]) / 2.0, (k_l[1] + k_r[1]) / 2.0)


def _approx_hip_width(pts, shoulder_w):
    """Estimate hip width.

    Hip landmarks are often unreliable (detected at thigh-gap when legs are
    together). Use knee midpoint × 0.85 as a proportional hip-width proxy.
    """
    km = _knee_mid(pts)
    knee_w = abs(pts[26][0] - pts[25][0])
    if knee_w < shoulder_w * 0.3:
        # Knees together — use shoulder-width × 0.9 as hip estimate
        return shoulder_w * 0.9
    # Hip is usually slightly narrower than knee span
    return max(knee_w * 0.85, shoulder_w * 0.85)


def compute_placement(pts, category, person_shape):
    """Compute where and how to place the cloth on the person.

    Returns ``(cx, cy, target_w, target_h, angle_deg)`` in pixel coords, where
    ``(cx, cy)`` is the cloth centre, ``target_w/h`` its size, and ``angle_deg``
    the torso tilt to rotate the cloth by.
    """
    h, w = person_shape[:2]

    sh_l = pts[_LANDMARK_L_SHOULDER]
    sh_r = pts[_LANDMARK_R_SHOULDER]
    ankle = _best_ankle(pts)
    shoulder_mid = _midpoint(sh_l, sh_r)
    shoulder_w = abs(sh_r[0] - sh_l[0])
    # body_length = shoulder-mid to ankle distance — reliable even when
    # hip landmarks are noisy (MediaPipe often detects thigh-gap as hips).
    body_length = ankle[1] - shoulder_mid[1]
    body_length = max(body_length, h * 0.05)
    shoulder_w = max(shoulder_w, w * 0.02)
    hip_w = _approx_hip_width(pts, shoulder_w)

    # Tilt of the shoulder line = torso tilt (rotate cloth to match posture).
    angle = math.degrees(math.atan2(sh_r[1] - sh_l[1], sh_r[0] - sh_l[0]))

    if category in ("top", "outerwear"):
        cx = shoulder_mid[0]
        cy = shoulder_mid[1] + body_length * 0.25
        factor = 1.25 if category == "top" else 1.40
        tw = shoulder_w * factor
        th = body_length * (0.75 if category == "top" else 0.95)
    elif category == "bottom":
        # Use knee midpoint as hip proxy (actual hip landmarks are unreliable).
        km = _knee_mid(pts)
        cx = km[0]
        cy = km[1] - body_length * 0.15
        tw = hip_w * 1.20
        th = body_length * 0.85
    elif category == "dress":
        cx = shoulder_mid[0]
        cy = shoulder_mid[1] + body_length * 0.55
        tw = shoulder_w * 1.20
        th = body_length * 1.65
    elif category == "shoes":
        # Use whichever foot is more visible; fall back to hip midpoint.
        foot_l = pts[_LANDMARK_L_FOOT]
        foot_r = pts[_LANDMARK_R_FOOT]
        if foot_l[2] >= foot_r[2]:
            foot = foot_l
        else:
            foot = foot_r
        cx = foot[0]
        cy = foot[1] - h * 0.01
        tw = shoulder_w * 0.55
        th = tw * 0.55
    else:  # accessories → head level
        nose = pts[_LANDMARK_NOSE]
        cx = nose[0]
        cy = nose[1] - h * 0.06
        tw = shoulder_w * 0.45
        th = tw

    return cx, cy, max(20, tw), max(20, th), angle



# ============ Cloth background removal ============

def make_cloth_mask(cloth_bgr):
    """Build a foreground (cloth) mask from a product photo.

    Strategy: Otsu thresholding on grayscale + largest-contour selection,
    followed by morphology cleanup. Falls back to HSV for very-light-producto
    images where Otsu picks the garment itself.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu auto-threshold — works for most white/light-gray studio backgrounds.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Also detect white (very bright) areas — product on pure white bg.
    hsv = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2HSV)
    white_hsv = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))

    # Combine: garment = Otsu mask OR white-HSV mask (union is more inclusive).
    mask = cv2.bitwise_or(thresh, white_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component = the garment.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        # stats[0] is background; pick the largest non-bg component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Guard: if largest area is < 2% of image, skip (avoid killing the garment on dark products).
        img_area = gray.shape[0] * gray.shape[1]
        if areas.max() > img_area * 0.02:
            largest = 1 + int(np.argmax(areas))
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        else:
            # Just apply morphology to the combined mask without area filtering.
            pass

    # GrabCut refinement for better edges (skip on huge images).
    mask = _grabcut_refine(cloth_bgr, mask)
    return mask


def _grabcut_refine(cloth_bgr, init_mask):
    """Refine the mask edges with GrabCut. No-op on failure or huge images."""
    import cv2
    import numpy as np

    h, w = cloth_bgr.shape[:2]
    # GrabCut allocates a few buffers of size HxW; cap to keep latency sane.
    if h * w > 800 * 800:
        return init_mask
    if init_mask.sum() == 0 or init_mask.sum() == 255 * h * w:
        return init_mask

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    gm = np.zeros((h, w), np.uint8)
    gm[init_mask == 255] = cv2.GC_FGD
    gm[init_mask == 0] = cv2.GC_BGD
    uncertain = (init_mask != 255) & (init_mask != 0)
    gm[uncertain] = cv2.GC_PR_FGD
    try:
        cv2.grabCut(cloth_bgr, gm, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except Exception as e:  # GrabCut can raise on near-empty masks
        logger.debug("GrabCut failed (%s); using HSV mask", e)
        return init_mask
    refined = np.where(
        (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    return refined


# ============ Overlay compositing ============

def _overlay_with_pose(person_bgr, cloth_bgr, pts, category) -> bool:
    """MediaPipe-driven placement: scale/rotate/feather onto the torso."""
    import cv2
    import numpy as np

    h, w = person_bgr.shape[:2]
    cx, cy, tw, th, angle = compute_placement(pts, category, person_bgr.shape)
    tw_i, th_i = int(tw), int(th)
    if tw_i < 10 or th_i < 10:
        return False

    # Resize cloth to target box and derive its mask.
    cloth_r = cv2.resize(cloth_bgr, (tw_i, th_i), interpolation=cv2.INTER_AREA)
    mask = make_cloth_mask(cloth_r)

    # Rotate cloth + mask together around the cloth centre.
    center = (tw_i / 2.0, th_i / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cloth_rot = cv2.warpAffine(
        cloth_r, M, (tw_i, th_i),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    mask_rot = cv2.warpAffine(
        mask, M, (tw_i, th_i),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Feather the alpha so the paste edge is soft.
    mask_soft = cv2.GaussianBlur(mask_rot, (5, 5), 0)
    alpha = (mask_soft.astype(np.float32) / 255.0)

    # Paste onto the person at (cx, cy), clipping to the image bounds.
    x0 = int(round(cx - tw_i / 2.0))
    y0 = int(round(cy - th_i / 2.0))
    x1, y1 = x0 + tw_i, y0 + th_i

    # Source/dest clipping
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    src_x1 = tw_i - max(0, x1 - w)
    src_y1 = th_i - max(0, y1 - h)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return False

    roi = person_bgr[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    cloth_roi = cloth_rot[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    a = alpha[src_y0:src_y1, src_x0:src_x1, None]
    blended = cloth_roi * a + roi * (1.0 - a)
    person_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
    return True


def _overlay_fixed_ratio(person_bgr, cloth_bgr, category) -> bool:
    """Fallback: fixed-ratio chest paste (previous behaviour).

    Used when MediaPipe is unavailable or no body was detected. Category is
    only used to nudge the vertical position a little.
    """
    import cv2
    import numpy as np

    h, w = person_bgr.shape[:2]
    target_w = max(40, int(w * 0.45))
    scale = target_w / cloth_bgr.shape[1]
    target_h = int(cloth_bgr.shape[0] * scale)
    cloth_r = cv2.resize(cloth_bgr, (target_w, target_h))

    hsv = cv2.cvtColor(cloth_r, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    mask = cv2.bitwise_not(white)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    x = (w - target_w) // 2
    # Shoes/dress sit a little lower than tops in the fallback.
    y_frac = {"top": 0.18, "outerwear": 0.16, "dress": 0.18,
              "bottom": 0.45, "shoes": 0.75, "accessories": 0.08}.get(category, 0.18)
    y = int(h * y_frac)
    y2 = min(y + target_h, h)
    x2 = min(x + target_w, w)
    mh, mw = y2 - y, x2 - x
    if mh <= 0 or mw <= 0:
        return False
    roi = person_bgr[y:y2, x:x2]
    m = (mask[:mh, :mw] > 0).astype(np.float32)
    for c in range(3):
        person_bgr[y:y2, x:x2, c] = (
            cloth_r[:mh, :mw, c] * m + roi[:, :, c] * (1 - m)
        ).astype(np.uint8)
    return True


def run_acgpn_tryon(person_path: str, cloth_path: str, output_path: str,
                    category: str = "top") -> str:
    """Run virtual try-on and write the result to ``output_path``.

    Returns the backend name used: ``"mediapipe"`` or ``"fallback"``.

    Args:
        person_path: Path to person image
        cloth_path: Path to clothing product image
        output_path: Where to save the composite
        category: top/bottom/dress/outerwear/shoes/accessories (default top)

    Returns:
        Backend name (raises never — on hard failure returns "fallback" after
        logging, so the caller can still report a best-effort image).
    """
    try:
        import cv2
        import numpy as np  # noqa: F401
    except ImportError as e:
        logger.error("OpenCV/numpy missing: %s", e)
        return "fallback"

    person = cv2.imread(person_path)
    cloth = cv2.imread(cloth_path)
    if person is None or cloth is None:
        logger.error("Could not read person/cloth image")
        return "fallback"

    # Cap very large inputs to keep pose detection + GrabCut fast.
    max_dim = 1024
    h, w = person.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        person = cv2.resize(person, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # Try the MediaPipe-driven placement first.
    pts = None
    mirrored = False
    try:
        result = detect_pose_landmarks(person)
        if result is not None:
            pts, mirrored = result
    except Exception as e:
        logger.warning("MediaPipe pose detection failed (%s); using fallback", e)

    backend = "fallback"
    ok = False
    if pts is not None:
        try:
            # Mirror the cloth so it aligns correctly on a mirrored person.
            cloth_vis = cv2.flip(cloth, 1) if mirrored else cloth
            ok = _overlay_with_pose(person, cloth_vis, pts, category)
            if ok:
                backend = "mediapipe"
        except Exception as e:
            logger.warning("Pose overlay failed (%s); using fallback", e)
            ok = False

    if not ok:
        _overlay_fixed_ratio(person, cloth, category)
        backend = "fallback"

    cv2.imwrite(output_path, person)
    logger.info("Try-on saved (%s backend): %s", backend, output_path)
    return backend


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ============ API Endpoints ============

@app.get("/")
async def root():
    """Health check."""
    backend = check_tryon_backend()
    has_pose = backend["mediapipe"]
    return {
        "status": "healthy",
        "service": "Virtual Try-On API",
        "backends": backend,
        "pose_aware": has_pose,
        "message": (
            "MediaPipe pose-aware try-on available"
            if has_pose else
            "mediapipe not installed; using fixed-ratio fallback"
        ),
    }


@app.get("/tryon/categories")
async def get_categories() -> List[str]:
    """Get available clothing categories."""
    return list(CLOTHING_DATABASE.keys())


@app.get("/tryon/clothing")
async def get_clothing(category: Optional[str] = None) -> List[ClothingItem]:
    """Get clothing items, optionally filtered by category."""
    if category:
        items = CLOTHING_DATABASE.get(category, [])
    else:
        items = []
        for items_list in CLOTHING_DATABASE.values():
            items.extend(items_list)

    return [
        ClothingItem(
            id=item["id"],
            name=item["name"],
            category=item["category"],
            image_url=f"/static/clothing/{item['image']}",
            thumbnail_url=f"/static/clothing/thumb_{item['image']}",
        )
        for item in items
    ]


@app.post("/tryon/upload/person")
async def upload_person_image(file: UploadFile = File(...)) -> dict:
    """Upload person image for try-on."""
    ext = _validated_ext(file.filename, file.content_type or "")

    # Generate unique ID
    file_id = str(uuid.uuid4())
    filename = f"person_{file_id}.{ext}"
    filepath = UPLOAD_DIR / filename

    # Save file with size limit
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    with open(filepath, "wb") as f:
        f.write(content)

    return {
        "image_id": file_id,
        "filename": filename,
        "url": f"/static/uploads/{filename}",
    }


@app.post("/tryon/upload/cloth")
async def upload_cloth_image(file: UploadFile = File(...)) -> dict:
    """Upload cloth image for try-on."""
    ext = _validated_ext(file.filename, file.content_type or "")

    file_id = str(uuid.uuid4())
    filename = f"cloth_{file_id}.{ext}"
    filepath = UPLOAD_DIR / filename

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    with open(filepath, "wb") as f:
        f.write(content)

    return {
        "image_id": file_id,
        "filename": filename,
        "url": f"/static/uploads/{filename}",
    }


@app.post("/tryon", response_model=TryOnResponse)
async def virtual_tryon(request: TryOnRequest) -> TryOnResponse:
    """Generate virtual try-on image.

    Args:
        request: person_image_id, cloth_image_id, optional category
                 (top/bottom/dress/outerwear/shoes/accessories, default top)
    """
    if request.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{request.category}'. "
                   f"Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )

    # Find image paths by exact id (no substring matching)
    person_file = _find_file(UPLOAD_DIR, "person", request.person_image_id)
    cloth_file = _find_file(UPLOAD_DIR, "cloth", request.cloth_image_id)

    if not person_file:
        raise HTTPException(status_code=404, detail="Person image not found")
    if not cloth_file:
        raise HTTPException(status_code=404, detail="Cloth image not found")

    result_id = str(uuid.uuid4())
    result_filename = f"result_{result_id}.jpg"
    result_path = RESULT_DIR / result_filename

    backend = run_acgpn_tryon(
        str(person_file),
        str(cloth_file),
        str(result_path),
        request.category,
    )

    if not result_path.exists():
        raise HTTPException(status_code=500, detail="Try-on generation failed")

    explanation = (
        f"Pose-aware overlay ({backend} backend) — the cloth was scaled to "
        f"your shoulder width, rotated to match your torso tilt, cut out from "
        f"its product background, and feathered onto your photo at the "
        f"'{request.category}' position."
    )

    return TryOnResponse(
        result_id=result_id,
        result_url=f"/static/results/{result_filename}",
        explanation=explanation,
        backend=backend,
    )


# ============ Demo endpoint (default model + 1 套 fixed clothes) ============

# Default resources (1 套固定,用于论文 demo / 初次访问)
DEMO_MODELS = {
    "male": STATIC_DIR / "models" / "model_male.jpg",
    "female": STATIC_DIR / "models" / "model_female.jpg",
}
DEMO_CLOTHES = {
    "top": STATIC_DIR / "models" / "tshirt_demo.jpg",
    "bottom": STATIC_DIR / "models" / "pants_demo.jpg",
}


@app.post("/tryon/demo")
async def tryon_demo(gender: str = "female", category: str = "top") -> dict:
    """生成默认模特 + 默认衣服的试穿图(无需用户上传,论文 demo 用)。

    Args:
        gender: male | female
        category: top | bottom

    Returns:
        {"result_url": "/static/results/...", "backend": "mediapipe/...",
         "model": "...jpg", "cloth": "...jpg"}
    """
    if gender not in DEMO_MODELS:
        raise HTTPException(status_code=400, detail=f"gender must be male|female")
    if category not in DEMO_CLOTHES:
        raise HTTPException(status_code=400, detail=f"category must be top|bottom")
    if not DEMO_MODELS[gender].exists():
        raise HTTPException(status_code=500, detail=f"default model image missing: {DEMO_MODELS[gender]}")
    if not DEMO_CLOTHES[category].exists():
        raise HTTPException(status_code=500, detail=f"default cloth missing: {DEMO_CLOTHES[category]}")

    result_id = str(uuid.uuid4())
    result_filename = f"demo_{gender}_{category}_{result_id}.jpg"
    result_path = RESULT_DIR / result_filename

    backend = run_acgpn_tryon(
        str(DEMO_MODELS[gender]),
        str(DEMO_CLOTHES[category]),
        str(result_path),
        category,
    )
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="demo try-on failed")

    return {
        "result_url": f"/static/results/{result_filename}",
        "backend": backend,
        "model_url": f"/static/models/{DEMO_MODELS[gender].name}",
        "cloth_url": f"/static/models/{DEMO_CLOTHES[category].name}",
        "gender": gender,
        "category": category,
    }


@app.get("/tryon/result/{result_id}")
async def get_tryon_result(result_id: str):
    """Get try-on result image."""
    result_file = _find_file(RESULT_DIR, "result", result_id)
    if result_file:
        return FileResponse(result_file)

    raise HTTPException(status_code=404, detail="Result not found")


# ============ Static Files ============

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
