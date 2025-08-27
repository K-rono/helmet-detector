# Intersection Helmet Detection (Demo)

This is a Streamlit demo for rider helmet detection at intersections. It uses a dummy detector now and exposes a pluggable interface for adding real models later (e.g., YOLO/VGG).

## Features
- Single-page Streamlit app (light theme)
- Pluggable detector interface: `HelmetDetector.predict(image) -> DetectionResult`
- Dummy detector returning a fixed decision and bounding box
- On-image annotation with bounding box and confidence

## Project Structure
```
.
├─ app/
│  ├─ __init__.py
│  └─ main.py               # Streamlit entrypoint
├─ detectors/
│  ├─ __init__.py
│  ├─ base.py               # Detector interfaces and DTOs
│  └─ dummy.py              # Dummy detector implementation
├─ utils/
│  └─ visualize.py          # Drawing boxes on images
├─ .streamlit/
│  └─ config.toml           # Light theme configuration
├─ requirements.txt
└─ README.md
```

## Setup (Conda)
```
conda create -n helmet-detector python=3.11 -y
conda activate helmet-detector
pip install -r requirements.txt
```

## Run
```
streamlit run app/main.py
```

## Using the app
- Upload a rider image (JPG, JPEG, PNG, BMP, WEBP)
- The demo will display a fixed decision and draw a sample bounding box
- Confidence is shown on a 0–100 scale

## Swapping in a real model later
Implement a class that extends `HelmetDetector` and wire it up in `app/main.py`:
```python
from detectors.base import HelmetDetector, DetectionResult, BoundingBox
from PIL import Image

class YoloHelmetDetector(HelmetDetector):
    def __init__(self, model_path: str):
        # load your model here
        ...

    def predict(self, image: Image.Image) -> DetectionResult:
        # run inference, build boxes
        boxes = [BoundingBox(x1=10, y1=10, x2=100, y2=100, label="helmet", score=0.95)]
        return DetectionResult(label="Wearing Helmet", confidence=95.0, boxes=boxes)
```
Then in `app/main.py` replace `DummyHelmetDetector()` with `YoloHelmetDetector(model_path)`.

## Notes
- This is a local demo with a generic image size cap handled by Streamlit.
- The UI assumes YOLO-like bounding boxes will be available when integrating the real model.
