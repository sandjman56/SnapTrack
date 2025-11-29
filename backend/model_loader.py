"""
Legacy model loader placeholder.

SnapTrack previously relied on a character-level CNN defined in this
module. The OCR stack has been simplified to use EasyOCR's word-level
recognition (see ``backend/ocr_easy.py``), so the CNN and its weight
loading have been removed. Keeping this module avoids import errors for
older scripts while clearly directing contributors to the new pipeline.
"""
from __future__ import annotations


def load_model():
    """Deprecated entrypoint maintained for backward compatibility."""

    raise RuntimeError(
        "The character CNN has been removed. Use ocr_easy.run_easyocr for OCR."
    )
