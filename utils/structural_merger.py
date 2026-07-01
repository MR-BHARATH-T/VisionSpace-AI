# utils/structural_merger.py
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class StructuralMerger:
    """Isolates building features (doors/windows) from furniture objects list and formats them."""

    def merge_structures(
        self,
        scenes: List[Dict[str, Any]],
        merged_objects: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Scans both the model-identified doors/windows lists and the objects list

        to cleanly classify building structures.

        Returns:
            doors: List of formatted doors
            windows: List of formatted windows
            furniture: List of furniture objects (with structures removed)
        """
        doors_list: List[Dict[str, Any]] = []
        windows_list: List[Dict[str, Any]] = []
        furniture_objects: List[Dict[str, Any]] = []

        # 1. Collect doors and windows already correctly placed in their respective arrays by individual scenes
        for scene in scenes:
            for d in scene.get("doors", []):
                if d and isinstance(d, dict) and d.get("id") not in [x.get("id") for x in doors_list]:
                    doors_list.append(dict(d))
            for w in scene.get("windows", []):
                if w and isinstance(w, dict) and w.get("id") not in [x.get("id") for x in windows_list]:
                    windows_list.append(dict(w))

        # 2. Inspect merged objects list and isolate misclassified building structures
        for obj in merged_objects:
            name = (obj.get("name") or "").lower()
            category = (obj.get("category") or "").lower()
            subcategory = (obj.get("subcategory") or "").lower()
            combined = f"{name} {category} {subcategory}"

            is_door = any(k in combined for k in ["door", "doorway", "entry", "entrance", "gate", "portal", "exit", "threshold"]) and not any(ex in combined for ex in ["indoor", "outdoor"])
            is_window = any(k in combined for k in ["window", "glass", "pane", "curtain", "drape", "blind", "shutter", "sill", "skylight"])

            if is_door:
                d_id = obj["id"]
                d = {
                    "id": d_id,
                    "type": obj.get("subcategory") or "swing",
                    "opens": "unknown",
                    "estimated_width_cm": obj.get("estimated_dimensions", {}).get("width_cm") or 100.0,
                    "estimated_height_cm": obj.get("estimated_dimensions", {}).get("height_cm") or 200.0,
                    "blocked": False,
                    "bbox": obj.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 100})
                }
                if d_id not in [x.get("id") for x in doors_list]:
                    doors_list.append(d)
            elif is_window:
                w_id = obj["id"]
                w = {
                    "id": w_id,
                    "type": obj.get("subcategory") or "sliding",
                    "estimated_width_cm": obj.get("estimated_dimensions", {}).get("width_cm") or 120.0,
                    "estimated_height_cm": obj.get("estimated_dimensions", {}).get("height_cm") or 120.0,
                    "natural_light_contribution": "high" if any(k in str(obj) for k in ["exposure", "light", "sun"]) else "medium",
                    "curtains": True if any(k in str(obj) for k in ["curtain", "drape", "curtains", "drapes"]) else False,
                    "blinds": True if any(k in str(obj) for k in ["blind", "blinds", "shutter", "shutters"]) else False,
                    "glass_type": "clear_glass",
                    "bbox": obj.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 100})
                }
                if w_id not in [x.get("id") for x in windows_list]:
                    windows_list.append(w)
            else:
                furniture_objects.append(obj)

        return doors_list, windows_list, furniture_objects
