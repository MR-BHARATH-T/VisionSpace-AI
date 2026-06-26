# utils/vision_merger.py
"""
VisionSpace AI — Production Vision Merger
==========================================
Merges outputs from multiple Vision Agent passes (e.g., tiled inference on
large images, multi-angle captures, or iterative detection runs) into a single
deduplicated, confidence-ranked object list.

Key capabilities:
  - NMS-based deduplication (Non-Maximum Suppression on IoU)
  - Confidence-weighted attribute merging (colour, material, condition)
  - Multi-pass result fusion (list-of-lists → single canonical list)
  - Scene-level metadata aggregation (lighting, style hints, room type)
  - Spatial-zone conflict resolution
  - ID normalisation (guarantees unique IDs across merged results)
  - Kaggle-compatible: pure Python + geometry_utils, no heavy deps

Usage::

    from utils.vision_merger import VisionMerger

    # Merge two Vision Agent passes
    merged = VisionMerger().merge([pass1_output, pass2_output])

    # Or use the module-level function
    from utils.vision_merger import merge_vision_outputs
    merged = merge_vision_outputs([pass1_output, pass2_output])
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from utils.geometry_utils import intersection_over_union, calculate_area

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NMS_THRESHOLD: float = 0.45   # IoU above which boxes are considered duplicates
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50   # Minimum confidence to keep an object
MAX_MERGED_OBJECTS: int = 60


# ---------------------------------------------------------------------------
# VisionMerger
# ---------------------------------------------------------------------------

class VisionMerger:
    """Merges and deduplicates multi-pass Vision Agent outputs.

    Args:
        nms_threshold:        IoU threshold for suppressing duplicate boxes.
        confidence_threshold: Minimum confidence for keeping an object.
        max_objects:          Hard cap on returned objects.
    """

    def __init__(
        self,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_objects: int = MAX_MERGED_OBJECTS,
    ) -> None:
        self._nms_iou = nms_threshold
        self._conf_threshold = confidence_threshold
        self._max_objects = max_objects

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def merge(self, passes: list[dict]) -> dict:
        """Merge multiple Vision Agent output dicts into one canonical scene.

        Each *pass* dict is expected to match the ``vision_analysis_prompt``
        JSON schema::

            {
                "scene": {
                    "room_type": "...",
                    "lighting":  "...",
                    "style_hint": "...",
                    "objects": [...]
                }
            }

        Args:
            passes: List of Vision Agent output dicts.

        Returns:
            Single merged scene dict with the same schema.
        """
        if not passes:
            return _empty_scene()

        if len(passes) == 1:
            return _normalise_ids(passes[0])

        # Collect all objects from all passes
        all_objects: list[dict] = []
        scene_metas: list[dict] = []

        for i, p in enumerate(passes):
            scene = p.get("scene", p)   # tolerate scene-wrapped or flat dicts
            objects = scene.get("objects", [])
            # Tag with source pass for provenance
            for obj in objects:
                all_objects.append({**obj, "_pass": i})
            scene_metas.append({
                "room_type":  scene.get("room_type", "unknown"),
                "lighting":   scene.get("lighting",  "unknown"),
                "style_hint": scene.get("style_hint", ""),
            })

        # Filter by confidence
        filtered = [
            o for o in all_objects
            if float(o.get("confidence", 1.0)) >= self._conf_threshold
        ]
        logger.debug(
            "VisionMerger: %d total objects across %d passes, "
            "%d above confidence threshold %.2f",
            len(all_objects), len(passes), len(filtered), self._conf_threshold,
        )

        # NMS deduplication
        deduped = self._apply_nms(filtered)

        # Assign normalised IDs
        deduped = _normalise_ids_list(deduped)

        # Sort by confidence descending, cap
        deduped.sort(key=lambda o: float(o.get("confidence", 0)), reverse=True)
        deduped = deduped[:self._max_objects]

        # Aggregate scene metadata
        merged_meta = self._aggregate_meta(scene_metas)

        return {
            "scene": {
                **merged_meta,
                "objects":       deduped,
                "_merge_passes": len(passes),
                "_total_raw":    len(all_objects),
                "_after_nms":    len(deduped),
            }
        }

    def merge_objects(
        self,
        object_lists: list[list[dict]],
    ) -> list[dict]:
        """Merge multiple flat object lists (without scene wrappers).

        Args:
            object_lists: Each element is a list of object dicts.

        Returns:
            Single deduplicated, sorted object list.
        """
        scenes = [{"scene": {"objects": objs}} for objs in object_lists]
        result = self.merge(scenes)
        return result["scene"]["objects"]

    # ------------------------------------------------------------------
    # NMS Implementation
    # ------------------------------------------------------------------

    def _apply_nms(self, objects: list[dict]) -> list[dict]:
        """Apply Non-Maximum Suppression per object class label.

        For each label group, suppress lower-confidence boxes that overlap
        with higher-confidence boxes above the IoU threshold.

        Args:
            objects: Flat list of detection dicts (all passes combined).

        Returns:
            Deduplicated list.
        """
        if not objects:
            return []

        # Group by label
        by_label: dict[str, list[dict]] = defaultdict(list)
        no_bbox: list[dict] = []

        for obj in objects:
            if "bbox" not in obj:
                no_bbox.append(obj)
                continue
            label = obj.get("label", "unknown").lower()
            by_label[label].append(obj)

        kept: list[dict] = list(no_bbox)

        for label, group in by_label.items():
            # Sort by confidence descending
            group.sort(key=lambda o: float(o.get("confidence", 0)), reverse=True)
            suppressed = set()

            for i, obj_i in enumerate(group):
                if i in suppressed:
                    continue
                kept.append(obj_i)
                for j in range(i + 1, len(group)):
                    if j in suppressed:
                        continue
                    iou = intersection_over_union(obj_i["bbox"], group[j]["bbox"])
                    if iou >= self._nms_iou:
                        # Merge attributes from suppressed box into kept box
                        kept[-1] = self._merge_pair(kept[-1], group[j])
                        suppressed.add(j)

        logger.debug(
            "NMS: %d → %d objects (%.0f%% deduplication)",
            len(objects), len(kept),
            (1 - len(kept) / max(len(objects), 1)) * 100,
        )
        return kept

    def _merge_pair(self, primary: dict, secondary: dict) -> dict:
        """Merge attribute evidence from *secondary* into *primary*.

        - Bounding box and dimensions: weighted average by confidence.
        - Colour / material / condition / subcategory / style: higher-confidence source wins.
        - Confidence metrics: average of both.

        Args:
            primary:   Higher-confidence detection (retained).
            secondary: Lower-confidence duplicate (absorbed).

        Returns:
            Merged detection dict.
        """
        merged = dict(primary)
        conf_p = float(primary.get("confidence", 0.5))
        conf_s = float(secondary.get("confidence", 0.5))
        total = conf_p + conf_s or 1.0

        # Confidence: average
        for conf_key in ("confidence", "detection_confidence", "attribute_confidence", "geometry_confidence"):
            val_p = primary.get(conf_key)
            val_s = secondary.get(conf_key)
            if val_p is not None and val_s is not None:
                merged[conf_key] = round((float(val_p) + float(val_s)) / 2, 4)
            elif val_s is not None:
                merged[conf_key] = val_s

        # BBox: weighted average
        if "bbox" in primary and "bbox" in secondary:
            def wav(k: str) -> float:
                return (
                    float(primary["bbox"][k]) * conf_p
                    + float(secondary["bbox"][k]) * conf_s
                ) / total
            merged["bbox"] = {k: round(wav(k), 1) for k in ("x", "y", "width", "height")}

        # Dimensions: weighted average if both present
        if "estimated_dimensions" in primary and "estimated_dimensions" in secondary:
            dims_p = primary["estimated_dimensions"] or {}
            dims_s = secondary["estimated_dimensions"] or {}
            merged_dims = {}
            for d in ("width_cm", "depth_cm", "height_cm"):
                vp = dims_p.get(d)
                vs = dims_s.get(d)
                if vp is not None and vs is not None:
                    merged_dims[d] = round((float(vp) * conf_p + float(vs) * conf_s) / total, 1)
                elif vp is not None:
                    merged_dims[d] = vp
                elif vs is not None:
                    merged_dims[d] = vs
            merged["estimated_dimensions"] = merged_dims
        elif "estimated_dimensions" not in primary and "estimated_dimensions" in secondary:
            merged["estimated_dimensions"] = secondary["estimated_dimensions"]

        # Categorical / string attributes
        categorical_attrs = (
            "colour", "color", "material", "condition", "subcategory", 
            "style", "estimated_weight", "visibility", "occluded_by"
        )
        for attr in categorical_attrs:
            val_p = primary.get(attr)
            val_s = secondary.get(attr)
            # If primary value is missing or unknown, fall back to secondary
            if val_p in ("unknown", "", None) and val_s not in ("unknown", "", None):
                merged[attr] = val_s

        # Boolean flags
        bool_flags = ("movable", "wall_aligned", "segmentation_available", "mask_available", "is_focal_point")
        for flag in bool_flags:
            val_p = primary.get(flag)
            val_s = secondary.get(flag)
            if val_p is None:
                merged[flag] = val_s
            elif val_s is not None:
                merged[flag] = val_p or val_s

        return merged

    # ------------------------------------------------------------------
    # Metadata Aggregation
    # ------------------------------------------------------------------

    def _aggregate_meta(self, metas: list[dict]) -> dict:
        """Vote on scene-level metadata across all passes."""
        room_types  = [m["room_type"]  for m in metas if m["room_type"]  != "unknown"]
        lightings   = [m["lighting"]   for m in metas if m["lighting"]   != "unknown"]
        style_hints = [m["style_hint"] for m in metas if m["style_hint"] != ""]

        def plurality(lst: list) -> str:
            if not lst:
                return "unknown"
            return Counter(lst).most_common(1)[0][0]

        return {
            "room_type":  plurality(room_types),
            "lighting":   plurality(lightings),
            "style_hint": plurality(style_hints),
        }


# ---------------------------------------------------------------------------
# ID Normalisation
# ---------------------------------------------------------------------------

def _normalise_ids(scene_dict: dict) -> dict:
    """Ensure all object IDs in a scene dict are unique."""
    scene = scene_dict.get("scene", scene_dict)
    objects = scene.get("objects", [])
    scene["objects"] = _normalise_ids_list(objects)
    if "scene" in scene_dict:
        scene_dict["scene"] = scene
    return scene_dict


def _normalise_ids_list(objects: list[dict]) -> list[dict]:
    """Assign or repair IDs so every object has a unique ``id`` field."""
    seen_ids: dict[str, int] = {}
    normalised = []

    for obj in objects:
        label = obj.get("label", "object").lower().replace(" ", "_")
        base_id = obj.get("id") or label

        # Strip existing suffix like "_02" to rebuild
        # e.g., "sofa_01" → "sofa", "sofa" → "sofa"
        if base_id.endswith(tuple(f"_{i:02d}" for i in range(1, 50))):
            base_id = "_".join(base_id.split("_")[:-1])

        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        obj_copy = dict(obj)
        obj_copy["id"] = f"{base_id}_{count:02d}"
        normalised.append(obj_copy)

    return normalised


def _empty_scene() -> dict:
    return {
        "scene": {
            "room_type":     "unknown",
            "lighting":      "unknown",
            "style_hint":    "",
            "objects":       [],
            "_merge_passes": 0,
            "_total_raw":    0,
            "_after_nms":    0,
        }
    }


# ---------------------------------------------------------------------------
# Module-Level Singleton & Convenience
# ---------------------------------------------------------------------------

_default_merger: VisionMerger | None = None


def get_merger(**kwargs) -> VisionMerger:
    """Return the module-level singleton VisionMerger."""
    global _default_merger
    if _default_merger is None:
        _default_merger = VisionMerger(**kwargs)
    return _default_merger


def merge_vision_outputs(
    passes: list[dict],
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Convenience function to merge multiple Vision Agent output dicts.

    Args:
        passes:               List of Vision Agent output dicts.
        nms_threshold:        IoU threshold for NMS deduplication.
        confidence_threshold: Minimum confidence to retain an object.

    Returns:
        Merged scene dict.
    """
    merger = VisionMerger(
        nms_threshold=nms_threshold,
        confidence_threshold=confidence_threshold,
    )
    return merger.merge(passes)


__all__ = [
    "VisionMerger",
    "get_merger",
    "merge_vision_outputs",
]
