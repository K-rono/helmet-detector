from __future__ import annotations

import io
import os
import sys
from typing import Optional

# Ensure project root is on sys.path so we can import sibling packages
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image

from detectors import DummyHelmetDetector
from utils.visualize import draw_detections


st.set_page_config(
	page_title="Intersection Helmet Detection",
	page_icon="🪖",
	layout="wide",
	initial_sidebar_state="collapsed",
)


def render_header():
	left, right = st.columns([0.7, 0.3])
	with left:
		st.title("Intersection Helmet Detection")
		st.caption(
			"Reducing rider fatality rates with AI-assisted helmet compliance checks at intersections."
		)
	with right:
		st.metric("Target Compliance", "95%", delta="+12% vs last month")


def render_uploader() -> Optional[Image.Image]:
	st.subheader("Upload Rider Image")
	file = st.file_uploader(
		"Drop a photo here (JPG, JPEG, PNG, BMP, WEBP)",
		type=["jpg", "jpeg", "png", "bmp", "webp"],
		accept_multiple_files=False,
		help="Image should contain a rider. This demo uses a dummy model.",
	)
	if file is None:
		return None
	bytes_data = file.read()
	image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
	st.success("Image loaded successfully.")
	return image


def render_prediction(image: Image.Image):
	detector = DummyHelmetDetector()
	result = detector.predict(image)

	annotated = draw_detections(image, result)

	c1, c2 = st.columns([0.55, 0.45])
	with c1:
		st.image(annotated, caption="Annotated Detection", use_column_width=True)

	with c2:
		st.subheader("Detection Result")
		st.markdown(
			f"**Decision:** {result.label}  ")
		st.progress(int(result.confidence))
		st.markdown(f"**Confidence:** {result.confidence:.0f} / 100")

		with st.expander("Details"):
			st.json(
				{
					"label": result.label,
					"confidence": result.confidence,
					"boxes": [
						{
							"x1": b.x1,
							"y1": b.y1,
							"x2": b.x2,
							"y2": b.y2,
							"label": b.label,
							"score": b.score,
						}
						for b in result.boxes
					],
				}
			)


def render_footer():
	st.markdown("---")
	st.caption(
		"Demo uses a dummy detector. Swap in YOLO/VGG via the pluggable detector interface."
	)


def main():
	render_header()
	image = render_uploader()
	if image is not None:
		render_prediction(image)
	render_footer()


if __name__ == "__main__":
	main()


