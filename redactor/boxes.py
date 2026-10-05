"""Manual redaction boxes: conversion between UI values and page-relative boxes."""
from .redact import Box

MIN_SIZE = 0.01  # ignore accidental clicks (<1% of the page in either direction)


def drag_to_box(drag: dict | None, page_index: int) -> Box | None:
    """Convert a drag from streamlit-image-coordinates (pixels of the displayed
    image) into a 0-1 page-relative box. Returns None for clicks/tiny drags."""
    if not drag or not drag.get("width") or not drag.get("height"):
        return None
    try:
        x1, y1, x2, y2 = (drag[k] for k in ("x1", "y1", "x2", "y2"))
    except KeyError:
        return None
    w, h = drag["width"], drag["height"]
    coords = [min(x1, x2) / w, min(y1, y2) / h, max(x1, x2) / w, max(y1, y2) / h]
    coords = [min(max(v, 0.0), 1.0) for v in coords]
    if coords[2] - coords[0] < MIN_SIZE or coords[3] - coords[1] < MIN_SIZE:
        return None
    return (page_index, *coords)


def box_to_row(box: Box) -> dict:
    """Editable table row: 1-based page, coordinates in percent."""
    pno, x0, y0, x1, y1 = box
    return {"page": pno + 1, "left %": round(x0 * 100, 1), "top %": round(y0 * 100, 1),
            "right %": round(x1 * 100, 1), "bottom %": round(y1 * 100, 1)}


def rows_to_boxes(rows) -> list[Box]:
    boxes = []
    for r in rows:
        try:
            boxes.append((int(r["page"]) - 1, r["left %"] / 100, r["top %"] / 100,
                          r["right %"] / 100, r["bottom %"] / 100))
        except (KeyError, TypeError, ValueError):
            continue  # half-filled row in the editor
    return boxes
