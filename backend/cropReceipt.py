"""
Receipt cropping utilities.

This module uses a line segment detector to locate the receipt region and
returns a cropped, deskewed version suitable for OCR. The logic is adapted
from the original top-level script while making it import friendly.
"""
from __future__ import annotations

import cv2
import numpy as np


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to increase contrast."""
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_receipt_lines(image: np.ndarray) -> np.ndarray:
    """Detect the receipt using a line segment detector and crop around it.

    Args:
        image: BGR image array loaded via OpenCV.

    Returns:
        Cropped receipt region. If detection fails, the original image is
        returned so downstream steps can continue gracefully.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to detect_receipt_lines")

    img0 = image.copy()
    h0, w0 = img0.shape[:2]
    scale = 1000 / max(h0, w0)
    img = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gamma_correction(img.copy(), 6), cv2.COLOR_BGR2GRAY)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Refine with Line Segment Detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    dlines = lsd.detect(edges)[0]

    # Create mask from detected lines
    mask = np.zeros_like(gray)
    if dlines is not None:
        for dline in dlines:
            x0, y0, x1, y1 = map(int, dline[0])
            cv2.line(mask, (x0, y0), (x1, y1), 255, 3)
    else:
        return img

    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # Find bounding box around detected region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    crop = img[y : y + h, x : x + w]
    return crop
