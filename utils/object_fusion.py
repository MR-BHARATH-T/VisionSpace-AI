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
        self.compatibility_groups = [
            {"bed", "mattress", "cot", "crib", "daybed"},
            {"chair", "sofa", "armchair", "stool", "couch", "seating", "bench", "recliner"},
            {"desk", "table", "nightstand", "counter", "desk_table"},
            {"dresser", "wardrobe", "closet", "cabinet", "cupboard", "drawer", "chest", "shelving", "bookcase"},
            {"lamp", "light", "sconce", "lighting", "chandelier", "fixture"}
        ]

    def are_compatible(self, label1: str, label2: str) -> bool:
        if label1 == label2:
            return True
        for group in self.compatibility_groups:
            if label1 in group and label2 in group:
                return True
                
        return False

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
        """Deduplicates objects globally across viewpoints using physical and semantic compatibility."""
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

        # Group by viewpoint index
        by_view = defaultdict(list)
        for obj in labeled_objs:
            by_view[obj.get("_view_source", 0)].append(obj)

        view_ids = sorted(by_view.keys())
        if not view_ids:
            return objects

        # Initialize global fusion list with the first viewpoint's candidates
        fused_list: List[Dict[str, Any]] = []
        for obj in by_view[view_ids[0]]:
            obj_copy = dict(obj)
            obj_copy["seen_in_views"] = [view_ids[0]]
            obj_copy["_match_scores"] = []
            fused_list.append(obj_copy)

        # Greedily fuse objects from subsequent viewpoint lists
        for v_idx in view_ids[1:]:
            view_objs = by_view[v_idx]
            for obj in view_objs:
                best_match_idx = -1
                best_score = -1.0
                label_o = obj["_normalized_label"]
                
                for k_idx, k_obj in enumerate(fused_list):
                    # Prevent merging candidates from the same viewpoint
                    if v_idx in k_obj.get("seen_in_views", []):
                        continue
                    
                    label_k = k_obj["_normalized_label"]
                    score = 0.0
                    
                    # 1. Category Compatibility check
                    if label_k == label_o:
                        score += 0.3
                    elif self.are_compatible(label_k, label_o):
                        score += 0.25
                    else:
                        score -= 0.5 # Penalty for mismatched categories

                    # 2. Word intersection check
                    name_k = k_obj.get("name", "").lower()
                    name_o = obj.get("name", "").lower()
                    words_k = set(name_k.split())
                    words_o = set(name_o.split())
                    if name_k == name_o:
                        score += 0.2
                    elif words_k.intersection(words_o):
                        score += 0.15

                    # 3. Material and color matching
                    if k_obj.get("material") == obj.get("material") and k_obj.get("material") is not None:
                        score += 0.1
                    if k_obj.get("color") == obj.get("color") and k_obj.get("color") is not None:
                        score += 0.1
                    
                    # 4. Proximity matching
                    pos_k = k_obj.get("position")
                    pos_o = obj.get("position")
                    if pos_k and pos_o:
                        dist = ((pos_k["x_cm"] - pos_o["x_cm"])**2 + (pos_k["y_cm"] - pos_o["y_cm"])**2)**0.5
                        if dist < 120:
                            score += 0.5
                        elif dist < 180:
                            score += 0.35
                        elif dist > 250:
                            score -= 0.6
                    else:
                        # Image coordinate fallback
                        center_k = k_obj["bbox"]["x"] + k_obj["bbox"]["width"] / 2
                        center_o = obj["bbox"]["x"] + obj["bbox"]["width"] / 2
                        if (center_k < 400 and center_o < 400) or (center_k >= 400 and center_o >= 400):
                            score += 0.3
                        
                    if score > best_score:
                        best_score = score
                        best_match_idx = k_idx
                
                if best_score >= 0.5 and best_match_idx != -1:
                    fused_list[best_match_idx] = self.merge_pair(fused_list[best_match_idx], obj)
                    fused_list[best_match_idx]["seen_in_views"].append(v_idx)
                    fused_list[best_match_idx]["_match_scores"].append(best_score)
                else:
                    obj_copy = dict(obj)
                    obj_copy["seen_in_views"] = [v_idx]
                    obj_copy["_match_scores"] = []
                    fused_list.append(obj_copy)

        # Singleton consolidation pass (for beds, dressers, wardrobes, desks, sofas)
        final_list: List[Dict[str, Any]] = []
        skip_indices = set()
        for i in range(len(fused_list)):
            if i in skip_indices:
                continue
            
            obj1 = fused_list[i]
            label1 = obj1["_normalized_label"]
            
            if label1 in self.unique_singletons:
                # Find all other instances of this singleton class and merge them all
                for j in range(i + 1, len(fused_list)):
                    if j in skip_indices:
                        continue
                    obj2 = fused_list[j]
                    label2 = obj2["_normalized_label"]
                    
                    if label2 == label1 or self.are_compatible(label1, label2):
                        obj1 = self.merge_pair(obj1, obj2)
                        obj1["seen_in_views"] = list(set(obj1.get("seen_in_views", []) + obj2.get("seen_in_views", [])))
                        obj1["_match_scores"] = obj1.get("_match_scores", []) + obj2.get("_match_scores", []) + [0.85]
                        skip_indices.add(j)
            
            final_list.append(obj1)

        # Formatting Output payload attributes
        kept: List[Dict[str, Any]] = []
        for k_obj in final_list:
            views = sorted(k_obj.get("seen_in_views", []))
            k_obj["merged_from_images"] = views
            
            # Match scores parsing or heuristic confidence based on view counts
            scores = k_obj.pop("_match_scores", [])
            if len(views) >= 2:
                k_obj["merge_confidence"] = round(min(0.95, 0.8 + 0.05 * len(views)), 2)
            else:
                k_obj["merge_confidence"] = 0.70
                
            k_obj.pop("seen_in_views", None)
            k_obj.pop("_normalized_label", None)
            k_obj.pop("_view_source", None)

            # Ensure all required confidence and source attributes are present
            if "estimated_dimensions" not in k_obj or not k_obj["estimated_dimensions"]:
                k_obj["estimated_dimensions"] = {"width_cm": 80, "depth_cm": 80, "height_cm": 80}
            
            if "dimension_confidence" not in k_obj["estimated_dimensions"] or k_obj["estimated_dimensions"]["dimension_confidence"] is None:
                k_obj["estimated_dimensions"]["dimension_confidence"] = 0.35
            if "measurement_source" not in k_obj["estimated_dimensions"] or k_obj["estimated_dimensions"]["measurement_source"] is None:
                k_obj["estimated_dimensions"]["measurement_source"] = "vision_estimation"

            if "room_position" not in k_obj or not k_obj["room_position"]:
                pos = k_obj.get("position", {"x_cm": 200, "y_cm": 250})
                k_obj["room_position"] = {
                    "x_cm": pos.get("x_cm", 200),
                    "y_cm": pos.get("y_cm", 250),
                    "rotation": k_obj.get("rotation_deg", 0)
                }
            if "depth_confidence" not in k_obj["room_position"] or k_obj["room_position"]["depth_confidence"] is None:
                k_obj["room_position"]["depth_confidence"] = 0.28
            if "position_source" not in k_obj["room_position"] or k_obj["room_position"]["position_source"] is None:
                k_obj["room_position"]["position_source"] = "vision_estimation"

            kept.append(k_obj)

        for o in no_bbox:
            o["merged_from_images"] = [o.get("_view_source", 0)]
            o["merge_confidence"] = 0.70
            o.pop("_normalized_label", None)
            o.pop("_view_source", None)
            
            if "estimated_dimensions" not in o or not o["estimated_dimensions"]:
                o["estimated_dimensions"] = {"width_cm": 80, "depth_cm": 80, "height_cm": 80}
            if "dimension_confidence" not in o["estimated_dimensions"] or o["estimated_dimensions"]["dimension_confidence"] is None:
                o["estimated_dimensions"]["dimension_confidence"] = 0.35
            if "measurement_source" not in o["estimated_dimensions"] or o["estimated_dimensions"]["measurement_source"] is None:
                o["estimated_dimensions"]["measurement_source"] = "vision_estimation"
                
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
        if "estimated_dimensions" in primary or "estimated_dimensions" in secondary:
            dims_p = primary.get("estimated_dimensions") or {}
            dims_s = secondary.get("estimated_dimensions") or {}
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
            
            # Preserve or compute dimension_confidence
            vp_conf = dims_p.get("dimension_confidence")
            vs_conf = dims_s.get("dimension_confidence")
            if vp_conf is not None and vs_conf is not None:
                merged_dims["dimension_confidence"] = round((float(vp_conf) + float(vs_conf)) / 2, 2)
            elif vp_conf is not None:
                merged_dims["dimension_confidence"] = vp_conf
            elif vs_conf is not None:
                merged_dims["dimension_confidence"] = vs_conf
            else:
                merged_dims["dimension_confidence"] = 0.35
                
            merged_dims["measurement_source"] = "vision_estimation"
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
