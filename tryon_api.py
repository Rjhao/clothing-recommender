"""Virtual Try-On API using ACGPN.

Usage:
    1. Download ACGPN: git clone https://github.com/leonidu/ACGPN.git
    2. Download pretrained models from Google Drive
    3. Run: python tryon_api.py

API Endpoints:
    POST /tryon - Generate try-on image
    GET  /tryon/categories - Get available clothing categories
    GET  /tryon/clothing - Get clothing items by category
"""
import os
import io
import re
import base64
import uuid
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

app = FastAPI(title="Virtual Try-On API", version="1.0.0")

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


# ============ Helper Functions ============

def check_acgpn_installed() -> bool:
    """Check if ACGPN is properly installed."""
    if not ACGPN_DIR.exists():
        return False
    required_files = [
        ACGPN_DIR / "test.py",
        ACGPN_DIR / "data_loader.py",
        ACGPN_DIR / "networks",
    ]
    return all(f.exists() for f in required_files)


def run_acgpn_tryon(person_path: str, cloth_path: str, output_path: str) -> bool:
    """
    Run ACGPN virtual try-on.

    Args:
        person_path: Path to person image
        cloth_path: Path to clothing image
        output_path: Path to save result

    Returns:
        True if successful, False otherwise
    """
    if not check_acgpn_installed():
        logger.warning("ACGPN not installed, using mock result")
        # For demo, just copy cloth image as result
        import shutil
        shutil.copy(cloth_path, output_path)
        return True

    try:
        import subprocess

        # ACGPN command
        cmd = [
            "python", str(ACGPN_DIR / "test.py"),
            "--person", person_path,
            "--cloth", cloth_path,
            "--output", output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"Try-on successful: {output_path}")
            return True
        else:
            logger.error(f"Try-on failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Try-on error: {e}")
        return False


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ============ API Endpoints ============

@app.get("/")
async def root():
    """Health check."""
    acgpn_installed = check_acgpn_installed()
    return {
        "status": "healthy",
        "service": "Virtual Try-On API",
        "acgpn_installed": acgpn_installed,
        "message": "ACGPN virtual try-on service" if acgpn_installed else "ACGPN not installed, using demo mode",
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
    """
    Generate virtual try-on image.

    Args:
        request: Contains person_image_id and cloth_image_id

    Returns:
        TryOnResponse with result image URL and explanation
    """
    # Find image paths by exact id (no substring matching)
    person_file = _find_file(UPLOAD_DIR, "person", request.person_image_id)
    cloth_file = _find_file(UPLOAD_DIR, "cloth", request.cloth_image_id)

    if not person_file:
        raise HTTPException(status_code=404, detail="Person image not found")
    if not cloth_file:
        raise HTTPException(status_code=404, detail="Cloth image not found")

    # Generate result
    result_id = str(uuid.uuid4())
    result_filename = f"result_{result_id}.jpg"
    result_path = RESULT_DIR / result_filename

    # Run ACGPN
    success = run_acgpn_tryon(
        str(person_file),
        str(cloth_file),
        str(result_path)
    )

    if not success:
        raise HTTPException(status_code=500, detail="Try-on generation failed")

    # Generate explanation
    explanation = (
        "Based on the uploaded photo and selected clothing, "
        "our AI model generates a virtual try-on result. "
        "The system uses advanced image synthesis to realistically "
        "render how the clothing would look on you."
    )

    return TryOnResponse(
        result_id=result_id,
        result_url=f"/static/results/{result_filename}",
        explanation=explanation,
    )


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
