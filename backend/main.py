from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cropReceipt import detect_receipt_lines
from extract_amount import parse_receipt
from ocr_easy import run_easy_ocr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECEIPTS_DIR = DATA_DIR / "receipts"
METADATA_FILE = DATA_DIR / "receipts.json"

for path in (DATA_DIR, RECEIPTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

if not METADATA_FILE.exists():
    METADATA_FILE.write_text("[]", encoding="utf-8")

app = FastAPI(title="SnapTrack OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/receipts", StaticFiles(directory=RECEIPTS_DIR), name="receipts")


class OCRBlock(BaseModel):
    bbox: List[List[float]]
    text: str
    confidence: float


class ReceiptItem(BaseModel):
    name: str
    price: float


class ReceiptResponse(BaseModel):
    store: str
    subtotal: Optional[float] = None
    total: Optional[float] = None
    taxes: List[float] = Field(default_factory=list)
    items: List[ReceiptItem]
    raw_text: List[str]
    blocks: List[OCRBlock]
    image_base64: str
    date: Optional[str] = None


class SaveReceiptRequest(BaseModel):
    subtotal: Optional[float] = None
    total: Optional[float] = None
    taxes: List[float] = Field(default_factory=list)
    items: List[ReceiptItem] = Field(default_factory=list)
    raw_text: List[str] | str
    image_base64: str
    store: Optional[str] = None
    date: Optional[str] = None


class HistoryResponse(BaseModel):
    receipts: List[dict]
    monthly_total: float
    month: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def decode_image(file_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def encode_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def load_receipts() -> list:
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def write_receipts(receipts: list) -> None:
    METADATA_FILE.write_text(json.dumps(receipts, indent=2), encoding="utf-8")


def normalize_raw_text(raw_text: List[str] | str) -> List[str]:
    if isinstance(raw_text, list):
        return raw_text
    return [line for line in raw_text.splitlines() if line.strip()]


def ensure_datetime_iso(date_str: Optional[str]) -> str:
    if date_str:
        try:
            return datetime.fromisoformat(date_str).isoformat()
        except ValueError:
            pass
    return datetime.utcnow().isoformat()


@app.post("/process_receipt", response_model=ReceiptResponse)
async def process_receipt(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        image = decode_image(contents)

        cropped = detect_receipt_lines(image)
        ocr_results = run_easy_ocr(cropped)

        parsed = parse_receipt(ocr_results)
        blocks = [
            {"bbox": bbox, "text": text, "confidence": float(conf)}
            for bbox, text, conf in ocr_results
        ]
        date = datetime.utcnow().date().isoformat()

        return {
            **parsed,
            "items": [ReceiptItem(**item) for item in parsed["items"]],
            "blocks": blocks,
            "image_base64": encode_base64(contents),
            "date": date,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/save_receipt")
async def save_receipt(payload: SaveReceiptRequest):
    try:
        receipts = load_receipts()
        receipt_id = str(uuid4())
        image_data = payload.image_base64
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        filename = f"{receipt_id}.png"
        image_path = RECEIPTS_DIR / filename
        image_path.write_bytes(image_bytes)

        entry = {
            "id": receipt_id,
            "date": ensure_datetime_iso(payload.date),
            "subtotal": payload.subtotal,
            "total": payload.total or payload.subtotal,
            "taxes": payload.taxes or [],
            "items": [item.dict() for item in payload.items],
            "image_path": f"/receipts/{filename}",
            "raw_text": normalize_raw_text(payload.raw_text),
            "store": payload.store or "Receipt",
        }
        receipts.append(entry)
        write_receipts(receipts)
        return entry
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history", response_model=HistoryResponse)
async def history():
    receipts = load_receipts()
    now = datetime.utcnow()
    month_key = now.strftime("%Y-%m")
    monthly_total = 0.0
    for receipt in receipts:
        date_str = receipt.get("date") or ""
        if date_str.startswith(month_key):
            monthly_total += float(receipt.get("total") or receipt.get("subtotal") or 0.0)

    return {
        "receipts": list(reversed(receipts)),
        "monthly_total": round(monthly_total, 2),
        "month": month_key,
    }
