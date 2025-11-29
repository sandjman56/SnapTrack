"""
FastAPI service for SnapTrack OCR.

Pipeline overview:
- Receipt cropping: backend.cropReceipt.detect_receipt_lines
- Word-level OCR: backend.ocr_easy.run_easyocr (EasyOCR)
- Parsing into items: parse_receipt_text utility below
- Returning JSON: /process_receipt route response

EasyOCR replaces the legacy character-level CNN and contour pipeline.
After the OpenCV-based cropping/deskew step, the API runs EasyOCR across
the entire cropped image to obtain newline-preserving text that feeds
directly into the receipt parser.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cropReceipt import detect_receipt_lines
from ocr_easy import run_easyocr

app = FastAPI(title="SnapTrack OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def parse_receipt_text(raw_text: str) -> Tuple[str, str, List[dict], float]:
    """Heuristically parse EasyOCR output into structured receipt data."""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    store = lines[0] if lines else ""

    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
    date_match = next(
        (m.group(1) for line in lines for m in [date_pattern.search(line)] if m),
        "",
    )

    total_pattern = re.compile(r"(Total|TOTAL|total)[: ]+\$?(\d+\.\d{2})")
    price_pattern = re.compile(r"\$?(\d+\.\d{2})")

    total_match = next(
        (m for line in lines for m in [total_pattern.search(line)] if m), None
    )
    total = float(total_match.group(2)) if total_match else 0.0

    items = []
    for line in lines[1:]:
        if total_pattern.search(line):
            continue
        price_match = price_pattern.search(line)
        if price_match:
            price = float(price_match.group(1))
            name = price_pattern.sub("", line).strip(" -:") or "Item"
            items.append({"name": name, "price": round(price, 2)})

    if not total and items:
        total = round(sum(item["price"] for item in items), 2)

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

        # Word-level OCR stage
        raw_text = run_easyocr(cropped)

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
