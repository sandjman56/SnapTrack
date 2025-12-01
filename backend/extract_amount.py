from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

BBox = List[Tuple[float, float]]
OCRResult = Tuple[BBox, str, float]

PRICE_REGEX = re.compile(r"\$?\d+\.\d{2}")
SUBTOTAL_KEYWORDS = ["subtotal", "sub total", "sub-total"]
TOTAL_KEYWORDS = [
    "total",
    "total purchase",
    "total amount",
    "purchase total",
    "total due",
    "amount due",
    "amount payable",
    "balance due",
    "balance to pay",
    "balance payable",
    "amount paid",
    "due today",
    "grand total",
    "total to pay",
    "amount owing",
]
TAX_KEYWORDS = ["tax", "sales tax", "vat", "tps", "tvq", "gst", "hst"]


def _bbox_bounds(bbox: BBox) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def price_from_text(text: str) -> Optional[float]:
    match = PRICE_REGEX.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def extract_amount_to_right(results: Iterable[OCRResult], ref_bbox: BBox) -> Optional[float]:
    """Find the nearest price value to the right of ref_bbox on the same horizontal band."""

    ref_x_min, ref_y_min, ref_x_max, ref_y_max = _bbox_bounds(ref_bbox)
    best_dx: Optional[float] = None
    best_amount: Optional[float] = None

    for bbox, text, _conf in results:
        x_min, y_min, x_max, y_max = _bbox_bounds(bbox)

        if x_min <= ref_x_max:
            continue

        vertical_overlap = min(ref_y_max, y_max) - max(ref_y_min, y_min)
        if vertical_overlap <= 0:
            continue

        amount = price_from_text(text)
        if amount is None:
            continue

        dx = x_min - ref_x_max
        if best_dx is None or dx < best_dx:
            best_dx = dx
            best_amount = amount

    return best_amount


def clean_text_lines(results: Iterable[OCRResult]) -> List[str]:
    return [text.strip() for _bbox, text, _conf in results if text and text.strip()]


def _contains_keyword(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def guess_store_name(lines: List[str]) -> str:
    for line in lines:
        if PRICE_REGEX.search(line):
            continue
        stripped = line.strip()
        if stripped and not stripped.isnumeric():
            return stripped
    return "Receipt"


def parse_line_items(lines: List[str]) -> List[dict]:
    items: List[dict] = []
    for line in lines:
        if _contains_keyword(line, SUBTOTAL_KEYWORDS + TOTAL_KEYWORDS + TAX_KEYWORDS):
            continue
        match = PRICE_REGEX.search(line)
        if not match:
            continue
        price = price_from_text(match.group(0))
        if price is None:
            continue
        name = line[: match.start()].strip(" :-") or "Item"
        items.append({"name": name, "price": round(price, 2)})
    return items


def _find_amount_by_keywords(results: Iterable[OCRResult], keywords: List[str]) -> Optional[float]:
    for bbox, text, _conf in results:
        lower = text.lower()
        if keywords is TOTAL_KEYWORDS and "subtotal" in lower:
            continue
        if not any(keyword in lower for keyword in keywords):
            continue
        amount = price_from_text(text)
        if amount is None:
            amount = extract_amount_to_right(results, bbox)
        if amount is not None:
            return round(amount, 2)
    return None


def collect_taxes(results: Iterable[OCRResult]) -> List[float]:
    taxes: List[float] = []
    for bbox, text, _conf in results:
        if not _contains_keyword(text, TAX_KEYWORDS):
            continue
        amount = price_from_text(text)
        if amount is None:
            amount = extract_amount_to_right(results, bbox)
        if amount is not None:
            taxes.append(round(amount, 2))
    return taxes


def parse_receipt(results: Iterable[OCRResult]) -> dict:
    results = list(results)
    lines = clean_text_lines(results)

    subtotal = _find_amount_by_keywords(results, SUBTOTAL_KEYWORDS)
    total = _find_amount_by_keywords(results, TOTAL_KEYWORDS)
    taxes = collect_taxes(results)

    if total is None:
        total = subtotal

    items = parse_line_items(lines)
    store = guess_store_name(lines)

    return {
        "store": store,
        "subtotal": subtotal,
        "total": total,
        "taxes": taxes,
        "items": items,
        "raw_text": lines,
    }
