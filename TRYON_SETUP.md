# Virtual Try-On Setup Guide

This document explains how to set up the ACGPN virtual try-on system.

## 1. Download ACGPN

```bash
# Clone ACGPN repository
cd D:/clothing-recommender
git clone https://github.com/leonidu/ACGPN.git
```

## 2. Download Pre-trained Models

Download from Google Drive:
- Link: https://drive.google.com/drive/folders/1T9B9e-G6E8bJ6g6L1L3v1l5Z5Z5Z5Z5

Or search for "ACGPN pretrained models" on Google.

Place the downloaded models in:
```
ACGPN/
├── checkpoint/
│   ├── GMM.pth
│   ├── TOM.pth
│   └── try_on.pth
```

## 3. Install Dependencies

```bash
# Install PyTorch (with CUDA support recommended)
pip install torch torchvision torchaudio

# Install other dependencies
pip install opencv-python pillow scipy numpy

# Install additional packages if needed
pip install scikit-image imageio-ffmpeg
```

## 4. Test ACGPN Standalone

```bash
cd ACGPN

# Download test images (optional)
# Place person.jpg and cloth.jpg in ACGPN/test/

# Run test
python test.py --person test/person.jpg --cloth test/cloth.jpg --output test/result.jpg
```

## 5. Run Virtual Try-On API

```bash
# From project root
cd D:/clothing-recommender

# Run the API
python tryon_api.py

# API will be available at http://localhost:8001
```

## 6. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/tryon/categories` | Get clothing categories |
| GET | `/tryon/clothing?category=top` | Get clothing items |
| POST | `/tryon/upload/person` | Upload person image |
| POST | `/tryon/upload/cloth` | Upload clothing image |
| POST | `/tryon` | Generate try-on result |

## 7. Alternative: VITON-HD

If ACGPN doesn't work well, try VITON-HD:

```bash
# Clone VITON-HD
git clone https://github.com/VITON/VITON-HD.git

# Follow their setup instructions
```

## 8. Troubleshooting

### Out of Memory
- Reduce image size (max 512x1024)
- Use smaller batch size

### Model download fails
- Use wget or curl with the direct link
- Check Google Drive permissions

### CUDA out of memory
```python
# Add to test.py
import torch
torch.cuda.empty_cache()
```

## Dataset Links

| Dataset | Link | Size |
|---------|------|------|
| VITON-HD | https://github.com/VITON/VITON-HD | ~3GB |
| ACGPN Dataset | https://drive.google.com/... | ~5GB |
| DeepFashion2 | https://github.com/switchablenorms/DeepFashion2 | ~3GB |

## Performance Tips

1. Use GPU with 8GB+ VRAM for real-time processing
2. Pre-process images to 512x1024 resolution
3. Cache intermediate results
4. Use batch processing for multiple try-ons
