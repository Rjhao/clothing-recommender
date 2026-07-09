# Virtual Try-On Setup Guide

本文档说明虚拟试穿服务的实际部署方案。**当前方案已弃用 ACGPN 全套流水线**，改用 **MediaPipe Pose（Tasks API）+ OpenCV 贴图**，CPU 可跑、Python 3.14 可装。

> 方案变更原因见 `README.md`「开发计划 → 第八阶段」。简言之：ACGPN 缺 GMM/TOM/SCHP/OpenPose 全套权重，且依赖老版 torch，Python 3.14 上配极易失败。

---

## 1. 环境要求

- Python 3.14（实测 3.14.6）
- 已装：`torch`（2.12.1+cpu，仅其他模块用到）、`numpy` 2.x、`scipy`
- CPU 即可运行（无 GPU 要求）

---

## 2. 安装依赖

### 2.1 处理 OpenCV 变体冲突

MediaPipe 依赖 `opencv-contrib-python`，它与 `opencv-python` **都提供 `cv2`、不能共存**。先卸载前者，装同版本的 contrib：

```bash
pip uninstall opencv-python -y
pip install "opencv-contrib-python==5.0.0.93"
```

> 验证：`python -c "import cv2; print(cv2.__version__)"` → 应为 `5.0.0`。

### 2.2 安装 MediaPipe

```bash
pip install mediapipe==0.10.35
```

mediapipe 0.10.35 的 Windows wheel 是 `py3-none-win_amd64.whl`（pure-Python + 预编译二进制），**对 Python 3.14 适用**。它会拉取以下依赖：

| 依赖 | 用途 |
|------|------|
| `absl-py` | MediaPipe 内部 |
| `matplotlib` | 绘图工具（drawing_utils） |
| `sounddevice` | 音频（本服务未用，但被导入） |
| `flatbuffers` | 模型序列化 |
| `opencv-contrib-python` | 图像处理 |
| `certifi` | 证书 |

> ⚠️ **`mediapipe.solutions` 在 0.10.35 的 win wheel 中不存在**（精简版）。本服务改用 **Tasks API**（`PoseLandmarker`），见 `tryon_api.py` 的 `_get_pose_detector()`。**不要**用 `mp.solutions.pose`。

### 2.3 验证导入

```bash
python -c "import mediapipe, cv2; print('mediapipe', mediapipe.__version__, 'cv2', cv2.__version__)"
```

应输出 `mediapipe 0.10.35 cv2 5.0.0`。

---

## 3. 下载 Pose 模型

下载 `pose_landmarker_lite.task`（5.5 MB）到 `static/models/`：

```bash
mkdir -p static/models
curl -L -o static/models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

> 模型文件随仓库提交（5.5MB），clone 后开箱即用，无需手动下载。仅当模型缺失时才需执行上述命令。
>
> 缺失时 `tryon_api` 启动检测仍通过，但首次试穿会因模型缺失回退到固定比例贴图（`fallback` backend）。

---

## 4. 测试图片

仓库已内置测试图，位于 `static/test/`：

- `person.jpg` — 真人照（433×650，竖向人像，MediaPipe 能检测到 33 个关键点）
- `cloth.jpg` — 服装产品图（白底）
- `result.jpg` — 参考结果

---

## 5. 启动服务

```bash
cd D:/clothing-recommender
python tryon_api.py
```

服务监听 `http://localhost:8001`。首次试穿时会加载 Pose 模型（约几百 ms，之后单例复用）。

健康检查：

```bash
curl http://localhost:8001/
# {"status":"healthy","backends":{"opencv":true,"mediapipe":true},"pose_aware":true,...}
```

`pose_aware: true` 表示 MediaPipe 可用；若为 `false`，服务仍可运行但会回退固定比例贴图。

---

## 6. API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 + 后端可用性 |
| GET | `/tryon/categories` | 服装分类列表 |
| GET | `/tryon/clothing?category=top` | 按分类取服装 |
| POST | `/tryon/upload/person` | 上传人像（jpg/png，≤10MB） |
| POST | `/tryon/upload/cloth` | 上传服装图 |
| POST | `/tryon` | 生成试穿结果 |
| GET | `/tryon/result/{result_id}` | 取结果图 |

### `/tryon` 请求体

```json
{
  "person_image_id": "<upload 返回的 image_id>",
  "cloth_image_id": "<upload 返回的 image_id>",
  "category": "top"
}
```

`category` 可选值：`top` / `bottom` / `dress` / `outerwear` / `shoes` / `accessories`（默认 `top`），决定服装贴在人体的位置。非法值返回 400。

### 贴图逻辑（`category` → 人体定位）

| category | 定位锚点 | 缩放基准 |
|----------|----------|----------|
| top / outerwear | 肩中点下方 | 肩宽 × 1.25/1.40 |
| bottom | 髋中点下方 | 髋宽 × 1.20 |
| dress | 肩到髋中段 | 肩宽 × 1.20，纵跨躯干 × 1.65 |
| shoes | 脚部 | 肩宽 × 0.55 |
| accessories | 鼻部上方 | 肩宽 × 0.45 |

服装按**肩连线倾角旋转**贴合躯干姿态；用 **HSV + GrabCut** 抠掉产品图白底；**alpha 高斯羽化**软化边缘。

### `/tryon` 响应

```json
{
  "result_id": "uuid",
  "result_url": "/static/results/result_xxx.jpg",
  "explanation": "...",
  "backend": "mediapipe"
}
```

`backend` 为 `mediapipe`（姿态感知）或 `fallback`（固定比例，未检测到人体时）。

---

## 7. 命令行验证

不启动服务，直接调用核心函数验证：

```bash
python -c "
import tryon_api
b = tryon_api.run_acgpn_tryon('static/test/person.jpg', 'static/test/cloth.jpg', 'static/test/_out.jpg', 'top')
print('backend:', b)
"
```

应输出 `backend: mediapipe`。

---

## 8. 故障排查

### `module 'mediapipe' has no attribute 'solutions'`

0.10.35 的 win wheel 不含 `solutions`。本服务已改用 Tasks API，**不应出现此错误**。若你自行扩展代码用了 `mp.solutions.pose`，请改用 `PoseLandmarker`（见 `tryon_api._get_pose_detector`）。

### `opencv-contrib-python` 与 `opencv-python` 冲突

二者都提供 `cv2`。卸载一个：

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install "opencv-contrib-python==5.0.0.93"
```

### `FileNotFoundError: Pose model not found at .../pose_landmarker_lite.task`

模型缺失。按第 3 节下载，或确认 `static/models/pose_landmarker_lite.task` 存在。

### 检测不到人体 → 走 fallback

某些图（纯色、无完整人像、人太小）MediaPipe 检测不到 pose，服务自动回退固定比例贴图。属正常行为，看响应 `backend` 字段确认。

### 性能

- 首次试穿加载模型约 0.5–1s（之后单例复用）
- 单次试穿 CPU 约 0.5–2s（视图片大小；超过 1024px 的图会先缩放）
- GrabCut 对 >800×800 的图自动跳过以控延迟

---

## 9. 历史：ACGPN（已弃用）

原计划用 [ACGPN](https://github.com/levidu/ACGPN) 全套流水线，但本仓库 `ACGPN/checkpoints/` 仅含 `label2city` refine 权重，缺 GMM/TOM/SCHP/OpenPose，且依赖老版 torch（Python 3.14 装不上）。**ACGPN 目录现已不参与试穿流程**，保留仅供查阅。如需全套 ACGPN，请用 Python 3.8 + 老版 torch 自行配置，与本服务无关。
