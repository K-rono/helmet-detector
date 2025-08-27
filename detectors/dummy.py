from __future__ import annotations

from typing import Tuple

from PIL import Image

from .base import HelmetDetector, DetectionResult, BoundingBox


class DummyHelmetDetector(HelmetDetector):
    """A deterministic placeholder detector.

    Returns a fixed decision (helmet present) with a single bounding box
    centered on the image. Confidence is fixed to 92.
    """

    def __init__(self, label: str = "Wearing Helmet", confidence: float = 92.0):
        self._label = label
        self._confidence = confidence

    def _center_box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        # Create a box ~50% of the shorter dimension, centered.
        size = int(min(width, height) * 0.5)
        cx, cy = width // 2, height // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(width - 1, x1 + size)
        y2 = min(height - 1, y1 + size)
        return x1, y1, x2, y2

    def predict(self, image: Image.Image) -> DetectionResult:
        width, height = image.size
        x1, y1, x2, y2 = self._center_box(width, height)
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label="helmet", score=self._confidence / 100.0)
        return DetectionResult(
            label=self._label,
            confidence=self._confidence,
            boxes=[box],
            raw={"note": "This is a dummy output. Replace with real model integration."},
        )


