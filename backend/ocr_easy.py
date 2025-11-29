"""
EasyOCR wrapper used by the SnapTrack backend.

The ``run_easyocr`` helper provides a single entrypoint for converting a
preprocessed receipt image into raw text with line breaks preserved. It
initializes a shared EasyOCR reader once at import time to avoid
repeated model downloads and to keep request latency low.
"""
from __future__ import annotations

import easyocr

reader = easyocr.Reader(["en"])


def run_easyocr(image) -> str:
    """Run EasyOCR on the given image and return newline-separated text."""

    return "\n".join([result[1] for result in reader.readtext(image)])
