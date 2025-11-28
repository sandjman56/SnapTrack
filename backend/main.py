"""
FastAPI service for SnapTrack OCR.

Pipeline overview:
- Receipt cropping: backend.cropReceipt.detect_receipt_lines
- Contour detection: backend.findcontours.preprocess & find_text_contours
- ML OCR inference: backend.model_loader.predict_character
- Text reconstruction: reconstruct_text utility below
- Parsing into items: parse_receipt_text utility below
- Returning JSON: /process_receipt route response
"""
from __future__ import annotations

import re
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .cropReceipt import detect_receipt_lines
from .findcontours import find_text_contours, preprocess
from .model_loader import load_model, predict_character

app = FastAPI(title="SnapTrack OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL, CHARSET = load_model()


class ReceiptResponse(BaseModel):
    store: str
    date: str
    items: List[dict]
    total: float
    raw_text: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def reconstruct_text(boxes: List[Tuple[int, int, int, int]], chars: List[str]) -> str:
    """Reconstruct text left-to-right, top-to-bottom from bounding boxes."""
    if not boxes or not chars:
        return ""

    lines = []
    current_line = []
    last_y = None
    for (x, y, w, h), ch in zip(boxes, chars):
        if last_y is None or abs(y - last_y) < max(h, 12):
            current_line.append((x, ch, w))
        else:
            lines.append(current_line)
            current_line = [(x, ch, w)]
        last_y = y
    if current_line:
        lines.append(current_line)

    reconstructed_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda v: v[0])
        line_text = ""
        for idx, (x, ch, w) in enumerate(line_sorted):
            if idx > 0:
                prev_x, _, prev_w = line_sorted[idx - 1]
                if x - (prev_x + prev_w) > 10:
                    line_text += " "
            line_text += ch
        reconstructed_lines.append(line_text)

    return "\n".join(reconstructed_lines)


def parse_receipt_text(raw_text: str) -> Tuple[str, str, List[dict], float]:
    """Heuristically parse the reconstructed text into structured fields."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    store = lines[0] if lines else ""

    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
    date_match = next((m.group(1) for line in lines for m in [date_pattern.search(line)] if m), "")

    price_pattern = re.compile(r"(\d+[\.,]?\d*)")
    items = []
    for line in lines[1:]:
        if "total" in line.lower():
            continue
        price_match = price_pattern.search(line)
        if price_match:
            price = float(price_match.group(1).replace(",", ""))
            name = line.replace(price_match.group(1), "").strip(" -:") or "Item"
            items.append({"name": name, "price": round(price, 2)})

    total_line = next((line for line in lines if "total" in line.lower()), "")
    total_match = price_pattern.search(total_line) if total_line else None
    total = float(total_match.group(1).replace(",", "")) if total_match else round(sum(i["price"] for i in items), 2)

    return store, date_match, items, total


@app.post("/process_receipt", response_model=ReceiptResponse)
async def process_receipt(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        image = decode_image(contents)

        # Receipt cropping stage
        cropped = detect_receipt_lines(image)

        # Contour detection stage
        binary = preprocess(cropped)
        boxes = find_text_contours(binary)

        # ML OCR inference stage
        chars = []
        for (x, y, w, h) in boxes:
            crop = cropped[y : y + h, x : x + w]
            chars.append(predict_character(MODEL, CHARSET, crop))

        # Text reconstruction stage
        raw_text = reconstruct_text(boxes, chars)

        # Parsing into structured JSON stage
        store, date, items, total = parse_receipt_text(raw_text)

        return {
            "store": store,
            "date": date,
            "items": items,
            "total": total,
            "raw_text": raw_text,
        }
    except Exception as exc:  # noqa: BLE001 broad to surface user-friendly error
        raise HTTPException(status_code=500, detail=str(exc))
