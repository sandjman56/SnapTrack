from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

BBox = List[Tuple[float, float]]
OCRResult = Tuple[BBox, str, float]

PRICE_REGEX = re.compile(r"\$?([0-9]{1,3}(?:[0-9]{3})*|[0-9]+)(?:[.,]\d{2})")


def _bbox_bounds(bbox: BBox) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _text_amount(text: str) -> Optional[float]:
    match = PRICE_REGEX.search(text)
    if not match:
        return None
    value = match.group(0).replace("$", "").replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def extract_amount_to_right(results: Iterable[OCRResult], ref_bbox: BBox) -> Optional[float]:
    """Find the nearest numeric value to the right of ref_bbox on the same line."""

    ref_x_min, ref_y_min, ref_x_max, ref_y_max = _bbox_bounds(ref_bbox)
    best_dx = None
    best_amount: Optional[float] = None

    for bbox, text, _conf in results:
        x_min, y_min, x_max, y_max = _bbox_bounds(bbox)

        # must be to the right
        if x_min <= ref_x_max:
            continue

        # vertical overlap check
        vertical_overlap = min(ref_y_max, y_max) - max(ref_y_min, y_min)
        if vertical_overlap <= 0:
            continue

        amount = _text_amount(text)
        if amount is None:
            continue

        dx = x_min - ref_x_max
        if best_dx is None or dx < best_dx:
            best_dx = dx
            best_amount = amount

    return best_amount


def parse_line_items(results: Iterable[OCRResult]) -> List[dict]:
    items: List[dict] = []
    for _bbox, text, _conf in results:
        lower = text.lower()
        if "total" in lower or "subtotal" in lower:
            continue
        amount = _text_amount(text)
        if amount is None:
            continue
        name = PRICE_REGEX.sub("", text).strip(" :-") or "Item"
        items.append({"name": name, "price": round(amount, 2)})
    return items


def guess_store_name(results: Iterable[OCRResult]) -> str:
    for _bbox, text, _conf in results:
        if _text_amount(text) is None and text.strip():
            return text.strip()
    return "Receipt"
