# utils/geometry_utils.py
"""
VisionSpace AI — Production Geometry & Spatial Utilities
=========================================================
Provides:
  - Bounding-box arithmetic (area, IoU, centre, normalise, scale)
  - Spatial relationship detection (overlap, containment, adjacency, distance)
  - Room layout analysis (zone assignment, focal-point detection, clearance checks)
  - Grid / heat-map generation for density visualisation
  - Depth-estimation helpers (ordinal depth from bounding-box heuristics)

All functions operate on plain Python dicts / lists — no heavy ML dependency
so they can run on CPU-only Kaggle kernels without GPU overhead.

BoundingBox format used throughout::

    {
        "x":      int | float,   # left edge in pixels (or normalised 0-1)
        "y":      int | float,   # top edge in pixels (or normalised 0-1)
        "width":  int | float,   # box width
        "height": int | float,   # box height
    }
"""

from __future__ import annotations

import logging
import math
from typing import Any

from utils.constants import (
    MIN_CLEARANCE_CM,
    ZONE_FURNITURE_MAP,
    FOCAL_POINT_BY_ROOM,
    ROOM_AREA_GUIDELINES_SQM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

BBox = dict[str, float]        # {"x", "y", "width", "height"}
Point = tuple[float, float]    # (x, y)


# ---------------------------------------------------------------------------
# Basic BoundingBox Arithmetic
# ---------------------------------------------------------------------------

def calculate_area(bbox: BBox) -> float:
    """Return the pixel area of *bbox*.

    Args:
        bbox: Dict with keys ``x``, ``y``, ``width``, ``height``.

    Returns:
        Non-negative float.  Returns 0.0 for degenerate boxes.
    """
    area = float(bbox["width"]) * float(bbox["height"])
    return max(0.0, area)


def bbox_centre(bbox: BBox) -> Point:
    """Return the centre point ``(cx, cy)`` of *bbox*."""
    cx = float(bbox["x"]) + float(bbox["width"]) / 2.0
    cy = float(bbox["y"]) + float(bbox["height"]) / 2.0
    return cx, cy


def bbox_corners(bbox: BBox) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` corner coordinates from a bbox dict."""
    x1 = float(bbox["x"])
    y1 = float(bbox["y"])
    x2 = x1 + float(bbox["width"])
    y2 = y1 + float(bbox["height"])
    return x1, y1, x2, y2


def normalise_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    """Return a copy of *bbox* with all coordinates scaled to [0, 1].

    Args:
        bbox:         Pixel-coordinate bbox.
        image_width:  Width of the reference image in pixels.
        image_height: Height of the reference image in pixels.

    Returns:
        New bbox dict with normalised float values.
    """
    return {
        "x":      float(bbox["x"])      / image_width,
        "y":      float(bbox["y"])      / image_height,
        "width":  float(bbox["width"])  / image_width,
        "height": float(bbox["height"]) / image_height,
    }


def denormalise_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    """Inverse of :func:`normalise_bbox` — scale [0,1] coords to pixels."""
    return {
        "x":      float(bbox["x"])      * image_width,
        "y":      float(bbox["y"])      * image_height,
        "width":  float(bbox["width"])  * image_width,
        "height": float(bbox["height"]) * image_height,
    }


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    """Uniformly scale *bbox* dimensions by *scale* (anchor = top-left)."""
    return {
        "x":      float(bbox["x"])      * scale,
        "y":      float(bbox["y"])      * scale,
        "width":  float(bbox["width"])  * scale,
        "height": float(bbox["height"]) * scale,
    }


def expand_bbox(bbox: BBox, margin: float) -> BBox:
    """Expand *bbox* by *margin* pixels on all four sides (can go negative).

    Args:
        bbox:   Source bounding box.
        margin: Pixels to add (positive) or subtract (negative) on each side.

    Returns:
        Expanded bbox; width and height are clamped to ≥ 0.
    """
    return {
        "x":      float(bbox["x"])      - margin,
        "y":      float(bbox["y"])      - margin,
        "width":  max(0.0, float(bbox["width"])  + 2 * margin),
        "height": max(0.0, float(bbox["height"]) + 2 * margin),
    }


# ---------------------------------------------------------------------------
# Intersection / IoU
# ---------------------------------------------------------------------------

def _intersection_area(a: BBox, b: BBox) -> float:
    """Return the intersection area of two bboxes (0 if no overlap)."""
    ax1, ay1, ax2, ay2 = bbox_corners(a)
    bx1, by1, bx2, by2 = bbox_corners(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def intersection_over_union(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bboxes.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if union is zero.
    """
    inter = _intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    union = calculate_area(a) + calculate_area(b) - inter
    return inter / union if union > 0 else 0.0


def boxes_overlap(a: BBox, b: BBox, threshold: float = 0.0) -> bool:
    """Return True if IoU(a, b) > *threshold*."""
    return intersection_over_union(a, b) > threshold


# ---------------------------------------------------------------------------
# Spatial Relationships
# ---------------------------------------------------------------------------

def euclidean_distance(a: BBox, b: BBox) -> float:
    """Return the Euclidean distance (pixels) between the centres of *a* and *b*."""
    (ax, ay) = bbox_centre(a)
    (bx, by) = bbox_centre(b)
    return math.hypot(ax - bx, ay - by)


def manhattan_distance(a: BBox, b: BBox) -> float:
    """Return the L1 (Manhattan) distance between the centres of *a* and *b*."""
    (ax, ay) = bbox_centre(a)
    (bx, by) = bbox_centre(b)
    return abs(ax - bx) + abs(ay - by)


def is_contained(inner: BBox, outer: BBox) -> bool:
    """Return True if *inner* bbox is fully contained within *outer* bbox."""
    ix1, iy1, ix2, iy2 = bbox_corners(inner)
    ox1, oy1, ox2, oy2 = bbox_corners(outer)
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def relative_position(a: BBox, b: BBox) -> dict[str, Any]:
    """Describe the spatial relationship of *a* relative to *b*.

    Returns a dict with::

        {
            "dx":        float,        # signed horizontal offset (cx_a - cx_b)
            "dy":        float,        # signed vertical offset (cy_a - cy_b)
            "distance":  float,        # Euclidean centre-to-centre distance
            "direction": str,          # compass-style: "left", "right", "above",
                                       # "below", "above-left", …, "overlapping"
            "iou":       float,
        }
    """
    (ax, ay) = bbox_centre(a)
    (bx, by) = bbox_centre(b)
    dx = ax - bx
    dy = ay - by
    dist = math.hypot(dx, dy)
    iou = intersection_over_union(a, b)

    if iou > 0.3:
        direction = "overlapping"
    else:
        h = "left" if dx < 0 else "right"
        v = "above" if dy < 0 else "below"
        if abs(dx) < 0.1 * dist:
            direction = v
        elif abs(dy) < 0.1 * dist:
            direction = h
        else:
            direction = f"{v}-{h}"

    return {
        "dx":        round(dx, 2),
        "dy":        round(dy, 2),
        "distance":  round(dist, 2),
        "direction": direction,
        "iou":       round(iou, 4),
    }


def are_adjacent(
    a: BBox,
    b: BBox,
    gap_threshold: float = 50.0,
) -> bool:
    """Return True if *a* and *b* are within *gap_threshold* pixels of each other.

    Checks the shortest edge-to-edge distance (not centre-to-centre).

    Args:
        a:             First bbox.
        b:             Second bbox.
        gap_threshold: Max pixel gap to be considered adjacent.
    """
    ax1, ay1, ax2, ay2 = bbox_corners(a)
    bx1, by1, bx2, by2 = bbox_corners(b)

    # Horizontal gap
    hgap = max(0.0, max(bx1 - ax2, ax1 - bx2))
    # Vertical gap
    vgap = max(0.0, max(by1 - ay2, ay1 - by2))

    # If both gaps are zero, boxes are overlapping → treat as adjacent
    if hgap == 0 and vgap == 0:
        return True
    if hgap == 0:
        return vgap <= gap_threshold
    if vgap == 0:
        return hgap <= gap_threshold
    # Corner-to-corner diagonal
    return math.hypot(hgap, vgap) <= gap_threshold


# ---------------------------------------------------------------------------
# Room Layout Analysis
# ---------------------------------------------------------------------------

def estimate_room_size(
    objects: list[dict],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Estimate relative room size from the spatial spread of detected objects.

    Uses the bounding box of all objects combined as a proxy for visible area.

    Args:
        objects:      List of detected objects, each with a ``bbox`` field.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Dict with ``coverage_ratio``, ``estimated_category`` (small/medium/large),
        and ``spread_bbox``.
    """
    if not objects:
        return {
            "coverage_ratio": 0.0,
            "estimated_category": "unknown",
            "spread_bbox": None,
        }

    xs = [obj["bbox"]["x"] for obj in objects if "bbox" in obj]
    ys = [obj["bbox"]["y"] for obj in objects if "bbox" in obj]
    x2s = [obj["bbox"]["x"] + obj["bbox"]["width"]  for obj in objects if "bbox" in obj]
    y2s = [obj["bbox"]["y"] + obj["bbox"]["height"] for obj in objects if "bbox" in obj]

    if not xs:
        return {"coverage_ratio": 0.0, "estimated_category": "unknown", "spread_bbox": None}

    spread = {
        "x":      min(xs),
        "y":      min(ys),
        "width":  max(x2s) - min(xs),
        "height": max(y2s) - min(ys),
    }
    spread_area = calculate_area(spread)
    image_area = image_width * image_height
    ratio = spread_area / image_area if image_area > 0 else 0.0

    if ratio < 0.35:
        category = "small"
    elif ratio < 0.65:
        category = "medium"
    else:
        category = "large"

    return {
        "coverage_ratio":       round(ratio, 4),
        "estimated_category":   category,
        "spread_bbox":          spread,
    }


def assign_spatial_zones(
    objects: list[dict],
    image_width: int,
    image_height: int,
    room_type: str = "living_room",
) -> list[dict]:
    """Assign a spatial zone label (top-left, top-right, …, centre) to each object.

    Divides the image into a 3×3 grid and classifies objects by where their
    centre falls.

    Args:
        objects:      List of detected objects with ``bbox`` and ``label`` fields.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.
        room_type:    Room category (used for zone naming hints).

    Returns:
        Copy of *objects* with an added ``spatial_zone`` field on each item.
    """
    w_third = image_width / 3
    h_third = image_height / 3

    _grid = {
        (0, 0): "top-left",     (1, 0): "top-centre",     (2, 0): "top-right",
        (0, 1): "middle-left",  (1, 1): "centre",          (2, 1): "middle-right",
        (0, 2): "bottom-left",  (1, 2): "bottom-centre",   (2, 2): "bottom-right",
    }

    annotated = []
    for obj in objects:
        obj_copy = dict(obj)
        if "bbox" in obj:
            cx, cy = bbox_centre(obj["bbox"])
            col = min(int(cx / w_third), 2)
            row = min(int(cy / h_third), 2)
            obj_copy["spatial_zone"] = _grid.get((col, row), "unknown")
        else:
            obj_copy["spatial_zone"] = "unknown"
        annotated.append(obj_copy)

    return annotated


def detect_focal_point(
    objects: list[dict],
    room_type: str = "living_room",
) -> dict[str, Any]:
    """Identify the likely focal point of a room from detected objects.

    Focal points are matched against the ``FOCAL_POINT_BY_ROOM`` registry,
    falling back to the largest object by area.

    Args:
        objects:   Detected object list (each has ``label`` and ``bbox``).
        room_type: Room category key.

    Returns:
        Dict with ``label``, ``bbox``, ``confidence`` (heuristic 0–1),
        ``method`` (``"registry_match"`` or ``"largest_object"``).
    """
    focal_candidates = [
        fp.replace("_", " ") for fp in FOCAL_POINT_BY_ROOM.get(room_type, [])
    ]

    for obj in objects:
        label = obj.get("label", "").lower().replace("_", " ")
        for fp in focal_candidates:
            if fp in label or label in fp:
                return {
                    "label":      obj.get("label"),
                    "bbox":       obj.get("bbox"),
                    "confidence": 0.85,
                    "method":     "registry_match",
                }

    # Fallback: largest visible object
    if objects:
        largest = max(
            (o for o in objects if "bbox" in o),
            key=lambda o: calculate_area(o["bbox"]),
            default=None,
        )
        if largest:
            return {
                "label":      largest.get("label"),
                "bbox":       largest.get("bbox"),
                "confidence": 0.55,
                "method":     "largest_object",
            }

    return {"label": None, "bbox": None, "confidence": 0.0, "method": "none"}


def check_clearances(
    objects: list[dict],
    pixels_per_cm: float = 3.0,
) -> list[dict]:
    """Check whether pairs of objects satisfy minimum clearance requirements.

    Args:
        objects:        List of objects with ``label`` and ``bbox``.
        pixels_per_cm:  Scale factor from cm to pixels (default 3 px/cm).

    Returns:
        List of violation dicts::

            {
                "object_a": str,
                "object_b": str,
                "rule":     str,
                "required_cm": float,
                "actual_cm":   float,
                "violation":   bool,
            }
    """
    violations = []

    # Only check rules we can infer from label matching
    rule_pairs = [
        ("sofa",    "coffee_table", "sofa_to_coffee"),
        ("sofa",    "tv",           "sofa_to_tv"),
        ("bed",     "wall",         "bed_sides"),
        ("chair",   "table",        "dining_chair_pull"),
    ]

    label_map = {obj.get("label", "").lower(): obj for obj in objects}

    for kw_a, kw_b, rule_key in rule_pairs:
        required_cm = MIN_CLEARANCE_CM.get(rule_key, 0)
        for label_a, obj_a in label_map.items():
            if kw_a not in label_a:
                continue
            for label_b, obj_b in label_map.items():
                if kw_b not in label_b:
                    continue
                dist_px = euclidean_distance(obj_a["bbox"], obj_b["bbox"])
                dist_cm = dist_px / pixels_per_cm
                violations.append({
                    "object_a":    label_a,
                    "object_b":    label_b,
                    "rule":        rule_key,
                    "required_cm": required_cm,
                    "actual_cm":   round(dist_cm, 1),
                    "violation":   dist_cm < required_cm,
                })

    return violations


# ---------------------------------------------------------------------------
# Density Heat-Map Grid
# ---------------------------------------------------------------------------

def build_density_grid(
    objects: list[dict],
    image_width: int,
    image_height: int,
    grid_cols: int = 8,
    grid_rows: int = 8,
) -> list[list[int]]:
    """Build a 2-D density grid counting objects per cell.

    Args:
        objects:      List of objects with ``bbox`` fields.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.
        grid_cols:    Number of horizontal cells.
        grid_rows:    Number of vertical cells.

    Returns:
        2-D list ``grid[row][col]`` with object counts per cell.
    """
    grid: list[list[int]] = [[0] * grid_cols for _ in range(grid_rows)]

    cell_w = image_width / grid_cols
    cell_h = image_height / grid_rows

    for obj in objects:
        if "bbox" not in obj:
            continue
        cx, cy = bbox_centre(obj["bbox"])
        col = min(int(cx / cell_w), grid_cols - 1)
        row = min(int(cy / cell_h), grid_rows - 1)
        grid[row][col] += 1

    return grid


# ---------------------------------------------------------------------------
# Depth Estimation Heuristics
# ---------------------------------------------------------------------------

def estimate_depth_order(
    objects: list[dict],
    image_height: int,
) -> list[dict]:
    """Estimate ordinal depth (closer/farther) using vertical position heuristics.

    Objects lower in the frame (larger ``y + height``) are typically closer
    in perspective photography.  Returns *objects* sorted nearest-first with
    a ``depth_rank`` field added (1 = closest).

    Args:
        objects:      List of objects with ``bbox`` fields.
        image_height: Image height in pixels.

    Returns:
        Sorted list of objects with ``depth_rank`` (int) and
        ``depth_score`` (float 0–1, higher = nearer) added.
    """
    scored = []
    for obj in objects:
        if "bbox" not in obj:
            scored.append({**obj, "depth_rank": 999, "depth_score": 0.0})
            continue
        _, _, _, y2 = bbox_corners(obj["bbox"])
        score = y2 / image_height   # 0 (top) → 1 (bottom = nearer)
        scored.append({**obj, "depth_score": round(score, 4)})

    scored.sort(key=lambda o: o.get("depth_score", 0), reverse=True)
    for rank, obj in enumerate(scored, start=1):
        obj["depth_rank"] = rank

    return scored


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Arithmetic
    "calculate_area",
    "bbox_centre",
    "bbox_corners",
    "normalise_bbox",
    "denormalise_bbox",
    "scale_bbox",
    "expand_bbox",
    # IoU / Overlap
    "intersection_over_union",
    "boxes_overlap",
    # Spatial relations
    "euclidean_distance",
    "manhattan_distance",
    "is_contained",
    "relative_position",
    "are_adjacent",
    # Room analysis
    "estimate_room_size",
    "assign_spatial_zones",
    "detect_focal_point",
    "check_clearances",
    # Utilities
    "build_density_grid",
    "estimate_depth_order",
]
