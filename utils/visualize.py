from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from detectors.base import DetectionResult, BoundingBox


def _color_for_label(label: str) -> Tuple[int, int, int]:
    if label.lower() == "helmet":
        return 34, 197, 94  # green-ish
    return 239, 68, 68  # red-ish


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]):
    x1, y1, x2, y2 = xy
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Pillow 10+: use textbbox to measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    rect = (x1, max(0, y1 - text_h - 2 * pad), x1 + text_w + 2 * pad, y1)
    draw.rectangle(rect, fill=color + (80,))
    draw.text((x1 + pad, rect[1] + pad), text, fill=(255, 255, 255), font=font)


def draw_detections(image: Image.Image, detections: DetectionResult) -> Image.Image:
    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in detections.boxes:
        color = _color_for_label(box.label)
        xy = (box.x1, box.y1, box.x2, box.y2)
        draw.rectangle(xy, outline=color + (255,), width=4)
        label_text = f"{box.label} {int(box.score * 100)}%"
        _draw_label(draw, xy, label_text, color)

    return Image.alpha_composite(annotated, overlay).convert("RGB")


