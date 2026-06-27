# utils/object_fusion.py
import logging
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class ObjectFusion:
    """Handles semantic object deduplication and multi-view attribute fusion."""

    def __init__(self, nms_threshold: float = 0.45) -> None:
        self.nms_threshold = nms_threshold
        # Semantic classes that typically exist as a single unique instance in a room
        self.unique_singletons = {"bed", "desk", "dresser", "wardrobe", "sofa"}

    def normalize_label(self, obj: Dict[str, Any]) -> str:
        """Helper to extract a clean semantic label from object fields."""
        name = (obj.get("name") or "").lower()
        category = (obj.get("category") or "").lower()
        subcategory = (obj.get("subcategory") or "").lower()
        label = (obj.get("label") or "").lower()
        
        combined = f"{name} {category} {subcategory} {label}"
        if "bed" in combined:
            return "bed"
        if "chair" in combined or "sofa" in combined or "seating" in combined:
            return "chair"
        if "desk" in combined or "table" in combined:
            return "desk"
        if "dresser" in combined or "wardrobe" in combined or "cabinet" in combined or "cupboard" in combined or "drawer" in combined:
            return "dresser"
        if "lamp" in combined or "light" in combined or "sconce" in combined or "lighting" in combined:
            return "lamp"
        if "door" in combined:
            return "door"
        if "window" in combined:
            return "window"
        if "plant" in combined or "flower" in combined:
            return "plant"
        if "painting" in combined or "frame" in combined or "art" in combined or "decor" in combined:
            return "decor"
            
        for val in (name, category, label):
            if val:
                return val.strip().replace(" ", "_")
        return "object"

    def fuse_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates objects across multiple viewpoints.

        - Unique singletons (Bed, Desk) are fused across views.
        - Repeatable elements (Lamps, Chairs) are matched based on location/features.
        """
        if not objects:
            return []

        no_bbox = []
        labeled_objs = []
        for obj in objects:
            label = self.normalize_label(obj)
            obj_copy = dict(obj)
            obj_copy["_normalized_label"] = label
            if "bbox" not in obj_copy:
                no_bbox.append(obj_copy)
            else:
                labeled_objs.append(obj_copy)

        by_label = defaultdict(list)
        for obj in labeled_objs:
            by_label[obj["_normalized_label"]].append(obj)

        kept: List[Dict[str, Any]] = []

        for label, group in by_label.items():
            if label in self.unique_singletons:
                # Merge all instances of this label into one canonical object
                group.sort(key=lambda o: float(o.get("confidence", 0)), reverse=True)
                canonical = dict(group[0])
                canonical["seen_in_views"] = [canonical.get("_view_source", 0)]
                
                for other in group[1:]:
                    canonical = self.merge_pair(canonical, other)
                    view_idx = other.get("_view_source", 0)
                    if view_idx not in canonical["seen_in_views"]:
                        canonical["seen_in_views"].append(view_idx)
                
                canonical.pop("_normalized_label", None)
                canonical.pop("_view_source", None)
                kept.append(canonical)
            else:
                # Repeatable multi-instance classes: match across different views greedily
                by_view = defaultdict(list)
                for obj in group:
                    by_view[obj.get("_view_source", 0)].append(obj)

                view_ids = sorted(by_view.keys())
                if not view_ids:
                    continue
                
                category_kept: List[Dict[str, Any]] = []
                for obj in by_view[view_ids[0]]:
                    obj_copy = dict(obj)
                    obj_copy["seen_in_views"] = [view_ids[0]]
                    category_kept.append(obj_copy)

                for v_idx in view_ids[1:]:
                    view_objs = by_view[v_idx]
                    for obj in view_objs:
                        best_match_idx = -1
                        best_score = -1.0
                        
                        for k_idx, k_obj in enumerate(category_kept):
                            if v_idx in k_obj.get("seen_in_views", []):
                                continue
                                
                            score = 0.0
                            if k_obj.get("name", "").lower() == obj.get("name", "").lower():
                                score += 0.2
                            if k_obj.get("material") == obj.get("material"):
                                score += 0.2
                            if k_obj.get("color") == obj.get("color"):
                                score += 0.2
                            
                            center_k = k_obj["bbox"]["x"] + k_obj["bbox"]["width"] / 2
                            center_o = obj["bbox"]["x"] + obj["bbox"]["width"] / 2
                            if (center_k < 400 and center_o < 400) or (center_k >= 400 and center_o >= 400):
                                score += 0.4
                                
                            if score > best_score:
                                best_score = score
                                best_match_idx = k_idx
                        
                        if best_score >= 0.5 and best_match_idx != -1:
                            category_kept[best_match_idx] = self.merge_pair(category_kept[best_match_idx], obj)
                            category_kept[best_match_idx]["seen_in_views"].append(v_idx)
                        else:
                            obj_copy = dict(obj)
                            obj_copy["seen_in_views"] = [v_idx]
                            category_kept.append(obj_copy)

                for k_obj in category_kept:
                    k_obj.pop("_normalized_label", None)
                    k_obj.pop("_view_source", None)
                    kept.append(k_obj)

        for o in no_bbox:
            o.pop("_normalized_label", None)
            o.pop("_view_source", None)
            kept.append(o)

        return kept

    def merge_pair(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge attribute evidence from secondary into primary.

        Prefers primary (highest confidence) values, falling back to secondary if missing.
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

        # Dimensions: weighted average
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

        # Categorical / string attributes
        categorical_attrs = (
            "colour", "color", "material", "condition", "subcategory", 
            "style", "estimated_weight", "visibility", "occluded_by"
        )
        for attr in categorical_attrs:
            val_p = primary.get(attr)
            val_s = secondary.get(attr)
            if val_p in ("unknown", "", None) and val_s not in ("unknown", "", None):
                merged[attr] = val_s

        # Boolean flags
        bool_flags = ("movable", "wall_aligned", "segmentation_available", "mask_available")
        for flag in bool_flags:
            val_p = primary.get(flag)
            val_s = secondary.get(flag)
            if val_p is None:
                merged[flag] = val_s
            elif val_s is not None:
                merged[flag] = val_p or val_s

        return merged
