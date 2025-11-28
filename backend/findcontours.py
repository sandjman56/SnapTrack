"""
Contour detection utilities.

This module preprocesses the receipt image, extracts text-like bounding boxes,
and offers a visualization helper. Functions are adapted from the original
standalone script with safer defaults for service use.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple

MIN_AREA = 20
MAX_AREA = 10000


def preprocess(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, enhance contrast, and binarize adaptively."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean


def find_text_contours(binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find and filter text contours by area and shape."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 10:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def visualize_boxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw bounding boxes on the image for debugging."""
    vis = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return vis
