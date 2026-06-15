# Face Detection Service

A face detection service built on [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) and [FastAPI](https://fastapi.tiangolo.com/). Supports three input modes: static image, video file, and real-time webcam stream. Exposes an HTTP API for image processing and a WebSocket endpoint for browser-based live detection.

---

## Project Structure

```
project_root/
├── src/
│   ├── main_api.py                # FastAPI application, HTTP and WebSocket endpoints
│   ├── detector.py                # PoliInputFaceDetector — core MediaPipe wrapper
│   ├── agent_face_detector.py     # Standalone detection scripts (image / video / stream)
│   ├── detection_visualization.py # Bounding box and keypoint drawing utilities
│   ├── config.py                  # Paths and environment config
│   └── logger.py                  # Logging configuration
├── models/
│   └── face_detector.tflite       # MediaPipe face detection model
├── templates/
│   └── index.html                 # Frontend for the WebSocket stream demo
├── .env
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- Webcam (for live stream mode only)

---

## Installation

```bash
git clone <repo-url>
cd <repo-dir>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the MediaPipe face detection model and place it in the `models/` folder:

```bash
wget -q -O models/face_detector.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

Create a `.env` file in the project root:

```env
MODEL_PATH=models/face_detector.tflite
```

---

## Running

### API Server

```bash
uvicorn src.main_api:app --reload
```

The server starts at `http://localhost:8000`. Open the browser to see the WebSocket stream demo page.

### Standalone Scripts

```python
from src.agent_face_detector import image_detection, video_detection, stream_detection

image_detection("path/to/image.jpg")
video_detection("path/to/video.mp4")
stream_detection()   # uses webcam (press D to stop)
```

---

## API Reference

### `GET /`

Returns the HTML page with the live stream demo.

---

### `POST /process-image`

Detects faces in an uploaded image and returns the annotated result.

**Request**

| Field  | Type           | Description          |
|--------|----------------|----------------------|
| `file` | `UploadFile`   | JPEG or PNG image    |

**Response**

- `200 OK` — JPEG image with bounding boxes and keypoints drawn
- Header `X-Detection-Count` — number of faces detected

---

## HTML page view

<img width="1912" height="1150" alt="HTML page view image" src="https://github.com/user-attachments/assets/84e64cc1-2743-4063-815a-5c831f950f62" />

