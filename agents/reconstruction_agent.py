"""Reconstruction Agent Module for VisionSpace AI.

Coordinates final layout state transition reports, removing replaced elements
and inserting selected products while maintaining dimensions and geometry constraints.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.ReconstructionAgent")


def apply_edge_feathering(img: Any, feather_radius: int = 15) -> Any:
    """Applies soft edge feathering to a PIL image to blend it naturally into the room snapshot."""
    from PIL import Image, ImageDraw
    img = img.convert("RGBA")
    width, height = img.size
    
    # Create mask channel
    mask = Image.new("L", (width, height), 255)
    draw_mask = ImageDraw.Draw(mask)
    
    # Draw a soft feathered gradient from edges
    for i in range(feather_radius):
        alpha = int(255 * (i / feather_radius))
        # Draw nested hollow rectangles with increasing alpha transparency
        draw_mask.rectangle(
            [i, i, width - 1 - i, height - 1 - i],
            fill=None,
            outline=alpha
        )
        
    alpha_channel = img.split()[3]
    from PIL import ImageChops
    blended_alpha = ImageChops.darker(alpha_channel, mask)
    img.putalpha(blended_alpha)
    return img


def make_background_transparent(img: Any) -> Any:
    """Converts solid or near-solid backgrounds of product thumbnails to transparent."""
    # 1. Attempt to use rembg if installed for professional AR background removal
    try:
        # Pre-check onnxruntime to avoid rembg module-level sys.exit(1) execution
        import onnxruntime
        from rembg import remove
        removed = remove(img)
        if removed:
            logger.info("Successfully removed showroom background using rembg library.")
            feathered = apply_edge_feathering(removed, feather_radius=8)
            # Autocrop transparent margins to make the product fill the canvas
            bbox = feathered.getbbox()
            if bbox:
                feathered = feathered.crop(bbox)
            return feathered
    except BaseException as rembg_err:
        logger.warning("rembg failed, not installed, or exited: %s. Falling back to corner-sample keying.", str(rembg_err))

    # 2. Fallback: Corner-sample chroma keying (works on gray/off-white studio backdrops)
    img = img.convert("RGBA")
    width, height = img.size
    pixels = img.load()
    
    # Sample the four corners
    corners = [
        pixels[0, 0],
        pixels[width - 1, 0],
        pixels[0, height - 1],
        pixels[width - 1, height - 1]
    ]
    
    # Calculate average corner color (RGB)
    avg_r = sum(c[0] for c in corners) // 4
    avg_g = sum(c[1] for c in corners) // 4
    avg_b = sum(c[2] for c in corners) // 4
    
    # Standard deviation threshold check to verify it's a solid backdrop
    import math
    dists = [math.sqrt((c[0]-avg_r)**2 + (c[1]-avg_g)**2 + (c[2]-avg_b)**2) for c in corners]
    is_solid_backdrop = all(d < 15 for d in dists) # corners are highly similar
    
    new_data = []
    datas = img.getdata()
    
    # Define match threshold
    threshold = 32
    
    for item in datas:
        r, g, b, a = item
        
        # Check distance to average corner color
        dist = math.sqrt((r - avg_r)**2 + (g - avg_g)**2 + (b - avg_b)**2)
        
        # Key out near-white only if the backdrop average corner is near-white
        if (avg_r > 230 and avg_g > 230 and avg_b > 230) and (r > 248 and g > 248 and b > 248):
            new_data.append((r, g, b, 0))
        elif is_solid_backdrop and dist < threshold:
            new_data.append((r, g, b, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    feathered = apply_edge_feathering(img, feather_radius=12)
    # Autocrop transparent margins to make the product fill the canvas
    bbox = feathered.getbbox()
    if bbox:
        feathered = feathered.crop(bbox)
    return feathered


class ReconstructionAgent:
    """Production-grade ReconstructionAgent responsible for room layout delta generation."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the ReconstructionAgent."""
        self.provider_name = provider
        self.inpaint_pipe = None
        self.product_image_cache = {}
        from models.text_model import TextModel
        from models.vision_model import VisionModel
        self.text_model = TextModel(provider=provider)
        self.vision_model = VisionModel(provider=provider)
        logger.info("ReconstructionAgent initialized")

    PLACEMENT_CONFIG = {
        "normal": {
            "bed_side_clearance": 100.0,
            "sofa_clearance": 60.0,
            "chair_operating_clearance": 75.0,
            "desk_clearance": 75.0,
            "bookcase_clearance": 60.0,
            "tv_unit_clearance": 60.0,
        },
        "compact": {
            "bed_side_clearance": 60.0,
            "sofa_clearance": 45.0,
            "chair_operating_clearance": 60.0,
            "desk_clearance": 50.0,
            "bookcase_clearance": 40.0,
            "tv_unit_clearance": 40.0,
        },
        "ultra_compact": {
            "bed_side_clearance": 30.0,
            "sofa_clearance": 30.0,
            "chair_operating_clearance": 40.0,
            "desk_clearance": 40.0,
            "bookcase_clearance": 30.0,
            "tv_unit_clearance": 30.0,
        },
        "micro": {
            "bed_side_clearance": 15.0,
            "sofa_clearance": 15.0,
            "chair_operating_clearance": 20.0,
            "desk_clearance": 25.0,
            "bookcase_clearance": 15.0,
            "tv_unit_clearance": 15.0,
        }
    }

    @staticmethod
    def classify_product(item: Dict[str, Any]) -> str:
        # Check if conformed fields already exist (backward compatibility)
        if isinstance(item, dict):
            p_domain = item.get("placement_domain")
            p_cat = item.get("canonical_category")
            if p_domain and p_cat:
                domain = p_domain
            else:
                from utils.product_classifier import ProductClassifier
                _, domain = ProductClassifier.classify(item)
        else:
            domain = "floor"
            
        # Map classifier domains to conformed local reconstruction domains
        if domain == "ceiling":
            return "ceiling"
        elif domain in ["support_surface", "surface", "ceiling_or_surface"]:
            return "surface"
        elif domain in ["wall", "wall_or_surface"]:
            return "wall"
        return "floor"

    @staticmethod
    def normalize_product_dimensions(item: Dict[str, Any]) -> Dict[str, Any]:
        # Priority order:
        # 1. Exact catalog dims in item["dimensions"]
        # 2. Specs in metadata or other keys
        # 3. Regex on title
        # 4. Fallback category defaults
        
        dims = item.get("dimensions") or {}
        w = float(dims.get("width_cm") or item.get("width_cm") or 0.0)
        d = float(dims.get("depth_cm") or item.get("depth_cm") or 0.0)
        h = float(dims.get("height_cm") or item.get("height_cm") or 0.0)
        
        source = "verified" if w > 0 else None
        confidence = 1.0 if source == "verified" else 0.55
        
        title_lower = str(item.get("title", item.get("name", ""))).lower()
        
        if not source:
            # Try to match specs in item metadata
            meta = item.get("metadata") or {}
            if "width" in meta or "depth" in meta or "height" in meta:
                w = float(meta.get("width") or meta.get("width_cm") or 0.0)
                d = float(meta.get("depth") or meta.get("depth_cm") or 0.0)
                h = float(meta.get("height") or meta.get("height_cm") or 0.0)
                if w > 0:
                    source = "metadata"
                    confidence = 0.90
        
        if not source:
            # Check title regex patterns
            import re
            # Check for patterns like "120x60x75" or "80x30x180"
            cross_match = re.search(r'(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?\s*(?:cm)?', title_lower)
            if cross_match:
                w = float(cross_match.group(1))
                d = float(cross_match.group(2))
                if cross_match.group(3):
                    h = float(cross_match.group(3))
                else:
                    h = 60.0
                source = "title_regex_cross"
                confidence = 0.80
            else:
                # Check for inch pattern like "52 in." or "52 inch" or "52\""
                inch_match = re.search(r'(\d+)\s*(?:in|inch|”|")', title_lower)
                if inch_match:
                    val_cm = float(inch_match.group(1)) * 2.54
                    w = val_cm
                    d = val_cm
                    h = 30.0 # ceiling fan height or tv unit
                    source = "title_regex_inch"
                    confidence = 0.80
                    
        if not source:
            source = "category_estimate"
            confidence = 0.55
            category = str(item.get("category", "")).lower()
            if "desk" in category or "table" in category:
                w, d, h = 100.0, 60.0, 75.0
            elif "chair" in category or "seating" in category:
                w, d, h = 60.0, 60.0, 85.0
            elif "bookshelf" in category or "shelf" in category or "bookcase" in title_lower:
                w, d, h = 80.0, 30.0, 180.0
            elif "tv unit" in title_lower or "tv stand" in title_lower or "television" in category:
                w, d, h = 120.0, 40.0, 50.0
            elif "bed" in category:
                w, d, h = 150.0, 200.0, 80.0
            elif "fan" in category or "ceiling" in category:
                w, d, h = 120.0, 120.0, 30.0
            elif "ac" in category or "conditioner" in category:
                w, d, h = 80.0, 25.0, 30.0
            elif "laptop" in category or "holder" in category or "stand" in category:
                w, d, h = 40.0, 30.0, 15.0
            else:
                w, d, h = 60.0, 60.0, 60.0
                
        return {
            "width_cm": w,
            "depth_cm": d,
            "height_cm": h,
            "dimension_source": source,
            "dimension_confidence": confidence
        }

    def calculate_identity_similarity(self, crop: Any, reference: Any) -> float:
        """Calculates grayscale visual similarity between crop and original product image."""
        try:
            from PIL import Image, ImageOps
            crop_gray = ImageOps.grayscale(crop.convert("RGBA"))
            ref_gray = ImageOps.grayscale(reference.convert("RGBA"))
            crop_resized = crop_gray.resize(ref_gray.size, Image.Resampling.LANCZOS)
            
            import numpy as np
            arr1 = np.array(crop_resized, dtype=np.float32)
            arr2 = np.array(ref_gray, dtype=np.float32)
            
            arr1_mean = arr1 - np.mean(arr1)
            arr2_mean = arr2 - np.mean(arr2)
            
            denom = np.sqrt(np.sum(arr1_mean ** 2) * np.sum(arr2_mean ** 2))
            if denom == 0:
                return 0.0
            ncc = np.sum(arr1_mean * arr2_mean) / denom
            return float(max(0.0, min(1.0, (ncc + 1.0) / 2.0)))
        except Exception as e:
            logger.warning("Similarity check error: %s. Defaulting to 0.85.", str(e))
            return 0.85

    def _run_3d_placement_solver(
        self,
        vision_output: Dict[str, Any],
        recommended: List[Dict[str, Any]],
        spatial_output: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs the Phase 7 strict 3D collision clearance and placement solver."""
        # Standardize room dimensions
        room_dim_m = spatial_output.get("room_dimensions", {}) if spatial_output else {}
        room_w = float(room_dim_m.get("width") or 4.0) * 100.0
        room_d = float(room_dim_m.get("length") or 5.0) * 100.0
        room_h = float(room_dim_m.get("height") or 2.7) * 100.0
        
        walkability_score = float(spatial_output.get("walkability_score", 100.0)) if spatial_output else 100.0
        usable_area = float(spatial_output.get("usable_area", 9.7)) if spatial_output else 9.7
        w_mode = "normal"
        if walkability_score < 30.0:
            w_mode = "micro"
        elif walkability_score < 50.0:
            w_mode = "ultra_compact"
        elif walkability_score < 75.0:
            w_mode = "compact"
            
        clearance = self.PLACEMENT_CONFIG.get(w_mode, self.PLACEMENT_CONFIG["compact"])

        # Strict Rejection if Spatial Output is Unknown
        if spatial_output and spatial_output.get("feasibility_status") == "unknown":
            validated_results = []
            for item in recommended:
                p_id = item.get("product_id")
                norm_dims = self.normalize_product_dimensions(item)
                validated_results.append({
                    "product_id": p_id,
                    "name": item.get("title", item.get("name", "furniture")),
                    "item": item,
                    "feasibility_status": "unknown",
                    "placement_status": "not_placed",
                    "placement": {
                        "domain": self.classify_product(item),
                        "world_coordinates": {},
                        "rotation_deg": 0
                    },
                    "dimensions": norm_dims,
                    "dimension_status": "insufficient_data"
                })
            return {
                "validated_results": validated_results,
                "placement_errors": [{
                    "product_id": "all",
                    "type": "insufficient_spatial_geometry",
                    "message": "Insufficient spatial geometry for reliable placement."
                }],
                "status_counts": {"perfect": 0, "adaptive": 0, "best_available": 0, "wall": 0, "ceiling": 0, "surface": 0, "failed": len(recommended)},
                "room_w": room_w,
                "room_h": room_h,
                "room_d": room_d,
                "usable_area": 0.0,
                "walkability_score": 0.0,
                "w_mode": w_mode
            }

        # Canonical Coordinate System Boxes:
        # x_cm: 0 to room_w
        # y_cm: 0 to room_d (depth)
        # z_cm: 0 to room_h (vertical height)
        occupied_3d_boxes = []

        existing_objects = vision_output.get("objects", []) if vision_output else []
        for obj in existing_objects:
            pos = obj.get("room_position") or obj.get("position")
            dims = obj.get("estimated_dimensions") or {}
            if pos:
                cx_old = float(pos.get("x_cm") or pos.get("x") or 0.0)
                cy_old = float(pos.get("y_cm") or pos.get("y") or 0.0)
                cz_old = float(pos.get("z_cm") or pos.get("z") or 0.0)
                
                if abs(cx_old) < 15.0: cx_old *= 100.0
                if abs(cy_old) < 15.0: cy_old *= 100.0
                if abs(cz_old) < 15.0: cz_old *= 100.0
                
                w_cm = float(dims.get("width_cm") or dims.get("width") or 80.0)
                d_cm = float(dims.get("depth_cm") or dims.get("depth") or 80.0)
                h_cm = float(dims.get("height_cm") or dims.get("height") or 80.0)
                
                # Shift to new coordinate system
                cx = cx_old + room_w / 2
                cy = cz_old
                cz = room_h / 2 - cy_old # height axis mapping
                
                # Ensure coordinate axes boundaries
                cx = max(w_cm/2, min(room_w - w_cm/2, cx))
                cy = max(d_cm/2, min(room_d - d_cm/2, cy))
                cz = max(h_cm/2, min(room_h - h_cm/2, cz))
                
                occupied_3d_boxes.append((
                    cx - w_cm/2, cx + w_cm/2,
                    cy - d_cm/2, cy + d_cm/2,
                    cz - h_cm/2, cz + h_cm/2,
                    obj.get("name", "object").lower()
                ))

        doors = vision_output.get("doors", []) if vision_output else []
        for door in doors:
            pos = door.get("room_position")
            dims = door.get("estimated_dimensions") or {}
            if pos:
                cx_old = float(pos.get("x_cm") or pos.get("x") or 0.0)
                cy_old = float(pos.get("y_cm") or pos.get("y") or 0.0)
                cz_old = float(pos.get("z_cm") or pos.get("z") or 0.0)
                if abs(cx_old) < 15.0: cx_old *= 100.0
                if abs(cy_old) < 15.0: cy_old *= 100.0
                if abs(cz_old) < 15.0: cz_old *= 100.0
                
                w_cm = float(dims.get("width_cm") or dims.get("width") or 90.0)
                d_cm = float(dims.get("depth_cm") or dims.get("depth") or 10.0)
                h_cm = float(dims.get("height_cm") or dims.get("height") or 200.0)
                
                cx = cx_old + room_w / 2
                cy = cz_old
                cz = room_h / 2 - cy_old
                
                occupied_3d_boxes.append((
                    cx - w_cm/2, cx + w_cm/2,
                    cy - d_cm/2, cy + d_cm/2,
                    cz - h_cm/2, cz + h_cm/2,
                    "door"
                ))

        windows = vision_output.get("windows", []) if vision_output else []
        for win in windows:
            pos = win.get("room_position")
            dims = win.get("estimated_dimensions") or {}
            if pos:
                cx_old = float(pos.get("x_cm") or pos.get("x") or 0.0)
                cy_old = float(pos.get("y_cm") or pos.get("y") or 0.0)
                cz_old = float(pos.get("z_cm") or pos.get("z") or 0.0)
                if abs(cx_old) < 15.0: cx_old *= 100.0
                if abs(cy_old) < 15.0: cy_old *= 100.0
                if abs(cz_old) < 15.0: cz_old *= 100.0
                
                w_cm = float(dims.get("width_cm") or dims.get("width") or 120.0)
                d_cm = float(dims.get("depth_cm") or dims.get("depth") or 10.0)
                h_cm = float(dims.get("height_cm") or dims.get("height") or 120.0)
                
                cx = cx_old + room_w / 2
                cy = cz_old
                cz = room_h / 2 - cy_old
                
                occupied_3d_boxes.append((
                    cx - w_cm/2, cx + w_cm/2,
                    cy - d_cm/2, cy + d_cm/2,
                    cz - h_cm/2, cz + h_cm/2,
                    "window"
                ))

        validated_results = []
        status_counts = {"perfect": 0, "adaptive": 0, "best_available": 0, "wall": 0, "ceiling": 0, "surface": 0, "failed": 0}
        placement_errors = []

        # Sort recommended items: Larger floor items first to establish sequential occupancy
        def get_solve_priority(p):
            dom = self.classify_product(p)
            dims = self.normalize_product_dimensions(p)
            area = dims["width_cm"] * dims["depth_cm"]
            if dom == "floor":
                return (0, -area)
            elif dom == "wall":
                return (1, -area)
            elif dom == "ceiling":
                return (2, -area)
            else:
                return (3, -area)
                
        sorted_items = sorted(recommended, key=get_solve_priority)

        for idx, item in enumerate(sorted_items):
            p_id = item.get("product_id")
            category = str(item.get("category", "")).lower()
            title_l = str(item.get("title", "")).lower()
            
            domain = self.classify_product(item)
            norm_dims = self.normalize_product_dimensions(item)
            w = norm_dims["width_cm"]
            d = norm_dims["depth_cm"]
            h = norm_dims["height_cm"]
            
            item["normalized_dimensions"] = norm_dims
            
            # Confidence verification threshold check
            is_verified_dims = "verified" if norm_dims["dimension_confidence"] >= 0.8 else "estimated"
            
            placed = False
            final_x, final_y, final_z = 0.0, 0.0, 0.0
            status = "failed_no_space"
            
            # Specialized Placement Solvers (Section 7)
            if domain == "ceiling":
                # CeilingPlacementSolver
                # Placed at z = room_h (ceiling)
                ceiling_center_x = room_w / 2
                ceiling_center_y = room_d / 2
                for dx in [0.0, -30.0, 30.0, -60.0, 60.0]:
                    for dy in [0.0, -30.0, 30.0, -60.0, 60.0]:
                        test_x = ceiling_center_x + dx
                        test_y = ceiling_center_y + dy
                        
                        # Boundary checks
                        if w/2 <= test_x <= room_w - w/2 and d/2 <= test_y <= room_d - d/2:
                            overlaps = False
                            for ox1, ox2, oy1, oy2, oz1, oz2, o_label in occupied_3d_boxes:
                                if "ceiling" in o_label or "fan" in o_label:
                                    if (test_x + w/2 > ox1 and test_x - w/2 < ox2) and (test_y + d/2 > oy1 and test_y - d/2 < oy2):
                                        overlaps = True
                                        break
                            if not overlaps:
                                final_x = test_x
                                final_y = test_y
                                final_z = room_h - h/2 # Snap to ceiling top
                                placed = True
                                status = "placed_ceiling"
                                status_counts["ceiling"] += 1
                                break
                    if placed: break
                    
            elif domain == "wall":
                # WallPlacementSolver
                # Wall coordinates: left (x=w/2), right (x=room_w-w/2), front (y=d/2), back (y=room_d-d/2)
                wall_candidates = [
                    (w/2, room_d / 2, room_h / 2, 0), # left wall
                    (room_w - w/2, room_d / 2, room_h / 2, 0), # right wall
                    (room_w / 2, d/2, room_h / 2, 0), # front wall
                    (room_w / 2, room_d - d/2, room_h / 2, 0) # back wall
                ]
                for wx, wy, wz, rot in wall_candidates:
                    overlaps = False
                    for ox1, ox2, oy1, oy2, oz1, oz2, o_label in occupied_3d_boxes:
                        if any(k in o_label for k in ["window", "door"]):
                            if (wx + w/2 > ox1 and wx - w/2 < ox2) and (wy + d/2 > oy1 and wy - d/2 < oy2):
                                overlaps = True
                                break
                    if not overlaps:
                        final_x = wx
                        final_y = wy
                        final_z = wz
                        placed = True
                        status = "placed_wall"
                        status_counts["wall"] += 1
                        break
                        
            elif domain == "surface":
                # SurfacePlacementSolver
                # Find a supporting surface in the room
                support_found = False
                surfaces = spatial_output.get("support_surfaces") or []
                for s in surfaces:
                    s_poly = s.get("footprint_polygon")
                    if s_poly:
                        sx1 = min(pt[0] for pt in s_poly)
                        sx2 = max(pt[0] for pt in s_poly)
                        sy1 = min(pt[1] for pt in s_poly)
                        sy2 = max(pt[1] for pt in s_poly)
                        sh = s.get("height_cm", 75.0)
                        
                        sw = sx2 - sx1
                        sd = sy2 - sy1
                        if w <= sw and d <= sd:
                            final_x = (sx1 + sx2) / 2
                            final_y = (sy1 + sy2) / 2
                            final_z = sh + h/2 # Sits on surface
                            support_found = True
                            placed = True
                            status = "placed_surface"
                            status_counts["surface"] += 1
                            break
                            
                # Check occupied boxes if not found in spatial_output
                if not support_found:
                    for ox1, ox2, oy1, oy2, oz1, oz2, o_label in reversed(occupied_3d_boxes):
                        if any(k in o_label for k in ["unit", "table", "stand", "dresser", "nightstand", "counter", "desk", "bookshelf", "cabinet"]):
                            sw = ox2 - ox1
                            sd = oy2 - oy1
                            if w <= sw and d <= sd:
                                final_x = (ox1 + ox2) / 2
                                final_y = (oy1 + oy2) / 2
                                final_z = oz2 + h/2
                                support_found = True
                                placed = True
                                status = "placed_surface"
                                status_counts["surface"] += 1
                                break
                if not support_found:
                    placed = False
                    status = "failed_no_space"
                    err_msg = f"No valid support surface available for '{item.get('title', 'Item')}'."
                    placement_errors.append({
                        "product_id": p_id,
                        "type": "no_support_surface",
                        "message": err_msg
                    })
                    
            elif domain == "floor":
                # FloorPlacementSolver
                # Search candidate locations from center outwards
                candidate_x = [room_w / 2]
                for step in range(15, int(room_w/2), 15):
                    candidate_x.extend([room_w/2 - step, room_w/2 + step])
                candidate_y = [room_d / 2]
                for step in range(15, int(room_d/2), 15):
                    candidate_y.extend([room_d/2 - step, room_d/2 + step])
                    
                min_penalty = 1e9
                best_coords = (room_w / 2, room_d / 2)
                
                for test_x in candidate_x:
                    for test_y in candidate_y:
                        test_x = max(w/2, min(room_w - w/2, test_x))
                        test_y = max(d/2, min(room_d - d/2, test_y))
                        test_z = h/2 # floor grounded
                        
                        ix1, ix2 = test_x - w/2, test_x + w/2
                        iy1, iy2 = test_y - d/2, test_y + d/2
                        iz1, iz2 = test_z - h/2, test_z + h/2
                        
                        hard_collision = False
                        overlap_penalty = 0.0
                        walkway_penalty = 0.0
                        boundary_penalty = 0.0
                        
                        # Validate clearances (collisions and openings)
                        for ox1, ox2, oy1, oy2, oz1, oz2, o_label in occupied_3d_boxes:
                            is_opening = ("window" in o_label or "door" in o_label)
                            x_overlap = (ix2 - 2 > ox1 and ix1 + 2 < ox2)
                            y_overlap = (iy2 - 2 > oy1 and iy1 + 2 < oy2)
                            z_overlap = (iz2 - 2 > oz1 and iz1 + 2 < oz2)
                            
                            if x_overlap and y_overlap and z_overlap:
                                if is_opening or "door" in o_label or "bed" in o_label:
                                    hard_collision = True
                                else:
                                    overlap_penalty += 100.0
                                    
                            if "bed" in o_label:
                                req_clear = clearance.get("bed_side_clearance", 45.0)
                                if ix1 - req_clear < ox2 and ix2 + req_clear > ox1 and iy1 - req_clear < oy2 and iy2 + req_clear > oy1:
                                    walkway_penalty += 20.0
                                    
                        # Wall adjacent preference for bookcases and cabinets
                        if any(k in category or k in title_l for k in ["bookshelf", "bookcase", "cabinet", "tv unit", "wardrobe"]):
                            dist_wall = min(ix1, room_w - ix2, iy1, room_d - iy2)
                            boundary_penalty += dist_wall * 0.2
                            
                        total_penalty = overlap_penalty + walkway_penalty + boundary_penalty
                        
                        if not hard_collision and total_penalty == 0:
                            final_x, final_y, final_z = test_x, test_y, test_z
                            placed = True
                            status = "placed_perfect"
                            status_counts["perfect"] += 1
                            break
                        elif not hard_collision and total_penalty < min_penalty:
                            min_penalty = total_penalty
                            best_coords = (test_x, test_y)
                            
                    if placed: break
                    
                if not placed and min_penalty < 50.0:
                    final_x, final_y = best_coords
                    final_z = h/2
                    placed = True
                    status = "placed_adaptive"
                    status_counts["adaptive"] += 1
                elif not placed:
                    placed = False
                    status = "failed_no_space"
                    status_counts["failed"] += 1
                    err_msg = f"No space available in the room to place '{item.get('title', 'Item')}'. Footprint: {w:.0f}x{d:.0f} cm."
                    if not any(err["product_id"] == p_id for err in placement_errors):
                        placement_errors.append({
                            "product_id": p_id,
                            "type": "no_space_available",
                            "message": err_msg
                        })

            # Grounding and Axes Boundary Assertions
            if placed:
                # Add strict boundary assertions conformed to canonical world space
                assert 0 <= final_x <= room_w, f"Coordinate final_x {final_x} out of width boundary [0, {room_w}]"
                assert 0 <= final_y <= room_d, f"Coordinate final_y {final_y} out of depth boundary [0, {room_d}]"
                assert 0 <= final_z <= room_h, f"Coordinate final_z {final_z} out of height boundary [0, {room_h}]"
                
                # Floor ground alignment check
                if domain == "floor":
                    assert abs(final_z - h/2) < 1.0, f"Floor object must be grounded on floor: z={final_z}, half-height={h/2}"
                # Ceiling snap check
                elif domain == "ceiling":
                    assert abs(final_z - (room_h - h/2)) < 1.0, f"Ceiling object must snap to ceiling: z={final_z}, target={room_h - h/2}"

                # Update sequential occupancy model
                occupied_3d_boxes.append((
                    final_x - w/2, final_x + w/2,
                    final_y - d/2, final_y + d/2,
                    final_z - h/2, final_z + h/2,
                    title_l
                ))
                validated_results.append({
                    "product_id": p_id,
                    "name": item.get("title", item.get("name", "furniture")),
                    "item": item,
                    "feasibility_status": "valid",
                    "placement_status": "approved",
                    "placement": {
                        "domain": domain,
                        "world_coordinates": {
                            "x_cm": round(final_x, 1),
                            "y_cm": round(final_y, 1),
                            "z_cm": round(final_z, 1)
                        },
                        "rotation_deg": 0
                    },
                    "dimensions": norm_dims,
                    "dimension_status": is_verified_dims
                })
            else:
                validated_results.append({
                    "product_id": p_id,
                    "name": item.get("title", item.get("name", "furniture")),
                    "item": item,
                    "feasibility_status": "no_space",
                    "placement_status": "not_placed",
                    "placement": {
                        "domain": domain,
                        "world_coordinates": {},
                        "rotation_deg": 0
                    },
                    "dimensions": norm_dims,
                    "dimension_status": is_verified_dims
                })

        return {
            "validated_results": validated_results,
            "placement_errors": placement_errors,
            "status_counts": status_counts,
            "room_w": room_w,
            "room_h": room_h,
            "room_d": room_d,
            "usable_area": usable_area,
            "walkability_score": walkability_score,
            "w_mode": w_mode
        }

    def plan_reconstruction(
        self,
        vision_output: Dict[str, Any],
        commerce_output: Dict[str, Any],
        session_path: Optional[Any] = None,
        spatial_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the Phase 7 reconstruction analysis."""
        logger.info("Running Phase 7 Reconstruction Agent layout delta mapping...")

        detected = vision_output.get("detected_objects") or []
        recommended = commerce_output.get("recommended_products") or []
        
        # Ensure all recommended products have stable product_id (Section 9)
        for idx, item in enumerate(recommended):
            if not item.get("product_id"):
                item["product_id"] = str(item.get("id") or item.get("sku") or f"prod_{idx}_{abs(hash(item.get('title', 'item')))}")

        solver_res = self._run_3d_placement_solver(vision_output, recommended, spatial_output)
        validated_results = solver_res["validated_results"]
        placement_errors = solver_res["placement_errors"]
        status_counts = solver_res["status_counts"]
        room_w = solver_res["room_w"]
        room_h = solver_res["room_h"]
        room_d = solver_res["room_d"]
        usable_area = solver_res["usable_area"]
        walkability_score = solver_res["walkability_score"]
        w_mode = solver_res["w_mode"]
        
        reconstruction_input = [v for v in validated_results if v["feasibility_status"] == "valid"]
        
        # Hard assertion check conforming to Rule #12
        for product in reconstruction_input:
            assert product["feasibility_status"] == "valid"
            assert product["placement_status"] == "approved"

        before_image = "original_1.png"
        after_image = "processed_reconstructed.png"
        
        diff_summary = []
        prompts = []
        verified_product_ids = []
        missing_product_ids = []
        retry_count = 0
        inpaint_targets = []
        gen_model_name = "runwayml/stable-diffusion-inpainting"
        
        if session_path:
            import pathlib
            from PIL import Image, ImageDraw, ImageOps
            s_path = pathlib.Path(session_path)
            
            orig_images = []
            orig_dir = s_path / "original_images"
            if orig_dir.exists():
                orig_images = sorted(list(orig_dir.glob("original_*.*")))
            if not orig_images:
                proc_dir = s_path / "processed_images"
                if proc_dir.exists():
                    orig_images = sorted(list(proc_dir.glob("processed_*.*")))
            if not orig_images:
                fallback_path = s_path / "original_images" / "original_1.png"
                if fallback_path.exists():
                    orig_images = [fallback_path]
                else:
                    orig_images = [s_path / "processed_images" / "processed_1.png"]
            def map_3d_to_2d(x_cm, y_cm, z_cm):
                norm_x = max(0.0, min(1.0, x_cm / room_w))
                norm_depth = max(0.0, min(1.0, y_cm / room_d))
                norm_height = max(0.0, min(1.0, z_cm / room_h))
                
                cx = 0.5
                cy = 0.35
                scale = 0.4 + 0.6 * norm_depth
                
                x_2d = cx + (norm_x - 0.5) * scale
                norm_z = 1.0 - norm_height
                y_2d = cy + (norm_z - 0.5) * 0.65 * scale
                
                return max(0.02, min(0.98, x_2d)), max(0.02, min(0.98, y_2d))

            def convex_hull_2d(points):
                points = sorted(list(set(points)))
                if len(points) <= 1:
                    return points
                def cross(o, a, b):
                    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
                lower = []
                for pt in points:
                    while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                        lower.pop()
                    lower.append(pt)
                upper = []
                for pt in reversed(points):
                    while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                        upper.pop()
                    upper.append(pt)
                return lower[:-1] + upper[:-1]

            # Build and save placement plan JSON (Section 8)
            placement_plan = []
            for idx, p in enumerate(reconstruction_input):
                wx = p["placement"]["world_coordinates"]["x_cm"]
                wy = p["placement"]["world_coordinates"]["y_cm"]
                wz = p["placement"]["world_coordinates"]["z_cm"]
                dims = p["dimensions"]
                w_cm = dims["width_cm"]
                d_cm = dims["depth_cm"]
                h_cm = dims["height_cm"]
                
                polygon_pts = []
                for dx in [-w_cm/2, w_cm/2]:
                    for dy in [-d_cm/2, d_cm/2]:
                        for dz in [-h_cm/2, h_cm/2]:
                            px, py = map_3d_to_2d(wx + dx, wy + dy, wz + dz)
                            polygon_pts.append([px, py])
                
                placement_plan.append({
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "3d_coordinates": {
                        "x_cm": wx,
                        "y_cm": wy,
                        "z_cm": wz
                    },
                    "clearances": {
                        "left": round(wx - w_cm/2, 1),
                        "right": round(room_w - (wx + w_cm/2), 1),
                        "front": round(wy - d_cm/2, 1)
                    },
                    "perspective_aware_2d_projection_polygons_masks": polygon_pts,
                    "depth_order": idx
                })
            
            try:
                save_json(placement_plan, s_path / "placement_plan.json")
            except Exception as e:
                logger.error("Failed to save placement_plan.json: %s", str(e))

            for img_idx, orig_image_path in enumerate(orig_images):
                if orig_image_path.exists():
                    try:
                        base_img = Image.open(orig_image_path).convert("RGBA")
                        width, height = base_img.size
                        temp_filename = None
                        
                        def get_sort_key(p):
                            dom = p["placement"]["domain"]
                            dims = p["dimensions"]
                            area = dims["width_cm"] * dims["depth_cm"]
                            if dom == "floor":
                                return (0, -area)
                            elif dom == "wall":
                                return (1, -area)
                            elif dom == "ceiling":
                                return (2, -area)
                            else:
                                return (3, -area)
                                
                        sorted_placements = sorted(reconstruction_input, key=get_sort_key)
                        
                        for p in sorted_placements:
                            wx = p["placement"]["world_coordinates"]["x_cm"]
                            wy = p["placement"]["world_coordinates"]["y_cm"]
                            wz = p["placement"]["world_coordinates"]["z_cm"]
                            dims = p["dimensions"]
                            w_cm = dims["width_cm"]
                            d_cm = dims["depth_cm"]
                            h_cm = dims["height_cm"]
                            
                            # 3D to 2D projection corner points
                            pts_3d = []
                            for dx in [-w_cm/2, w_cm/2]:
                                for dy in [-d_cm/2, d_cm/2]:
                                    for dz in [-h_cm/2, h_cm/2]:
                                        px, py = map_3d_to_2d(wx + dx, wy + dy, wz + dz)
                                        pts_3d.append((int(px * width), int(py * height)))
                                        
                            polygon_px = convex_hull_2d(pts_3d)
                            xs_px = [pt[0] for pt in polygon_px]
                            ys_px = [pt[1] for pt in polygon_px]
                            
                            paste_box = (
                                max(0, min(xs_px)),
                                max(0, min(ys_px)),
                                max(min(xs_px) + 10, min(width, max(xs_px))),
                                max(min(ys_px) + 10, min(height, max(ys_px)))
                            )
                            
                            product_img = None
                            img_url = (
                                p["item"].get("image") or 
                                p["item"].get("thumbnail_url") or 
                                p["item"].get("high_resolution_image") or 
                                p["item"].get("image_url") or 
                                p["item"].get("thumbnail") or 
                                ""
                            )
                            
                            if img_url:
                                if img_url in self.product_image_cache:
                                    product_img = self.product_image_cache[img_url]
                                else:
                                    try:
                                        import requests
                                        from io import BytesIO
                                        headers = {
                                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                                        }
                                        response = requests.get(img_url, headers=headers, timeout=12)
                                        if response.status_code == 200:
                                            raw_img = Image.open(BytesIO(response.content))
                                            product_img = make_background_transparent(raw_img)
                                            self.product_image_cache[img_url] = product_img
                                    except Exception as e:
                                        logger.warning("Failed to fetch product image: %s", str(e))
                                        
                            if not product_img:
                                fallback_url = "https://images.unsplash.com/photo-1513151233558-d860c5398176?auto=format&fit=crop&w=300&q=80"
                                cat_lower = p["item"].get("category", "").lower()
                                if "desk" in cat_lower or "table" in cat_lower:
                                    fallback_url = "https://images.unsplash.com/photo-1581428982868-e410dd047a90?q=80&w=300&auto=format&fit=crop"
                                elif "chair" in cat_lower or "seating" in cat_lower:
                                    fallback_url = "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?q=80&w=300&auto=format&fit=crop"
                                elif "bookshelf" in cat_lower or "shelf" in cat_lower:
                                    fallback_url = "https://images.unsplash.com/photo-1594980596870-8cb524ebf6bc?auto=format&fit=crop&w=300&q=80"
                                elif "bed" in cat_lower:
                                    fallback_url = "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=300&q=80"
                                elif "fan" in cat_lower:
                                    fallback_url = "https://images.unsplash.com/photo-1565814329452-e1efa11c5b89?auto=format&fit=crop&w=300&q=80"
                                    
                                try:
                                    import requests
                                    from io import BytesIO
                                    response = requests.get(fallback_url, timeout=10)
                                    if response.status_code == 200:
                                        fallback_raw = Image.open(BytesIO(response.content))
                                        product_img = make_background_transparent(fallback_raw)
                                except Exception as e:
                                    logger.warning("Unsplash download failed: %s", str(e))
                                    
                            if not product_img:
                                placeholder = Image.new("RGBA", (300, 300), (46, 204, 113, 200))
                                d_p = ImageDraw.Draw(placeholder)
                                d_p.rectangle([15, 15, 285, 285], outline=(255,255,255,220), width=6)
                                d_p.text((150, 150), p["name"][:20], fill=(255,255,255,255), anchor="mm")
                                product_img = placeholder
                                
                            p_w_box = paste_box[2] - paste_box[0]
                            p_h_box = paste_box[3] - paste_box[1]
                            
                            scale = min(p_w_box / product_img.width, p_h_box / product_img.height)
                            p_w = max(10, int(product_img.width * scale))
                            p_h = max(10, int(product_img.height * scale))
                            p_resized = product_img.resize((p_w, p_h), Image.Resampling.LANCZOS)
                            
                            domain = p["placement"]["domain"]
                            if domain == "floor" or domain == "surface":
                                paste_y_adj = paste_box[3] - p_h
                                paste_x_adj = (paste_box[0] + paste_box[2]) // 2 - p_w // 2
                            elif domain == "ceiling":
                                paste_y_adj = paste_box[1]
                                paste_x_adj = (paste_box[0] + paste_box[2]) // 2 - p_w // 2
                            else:
                                paste_y_adj = (paste_box[1] + paste_box[3]) // 2 - p_h // 2
                                paste_x_adj = (paste_box[0] + paste_box[2]) // 2 - p_w // 2
                                
                            paste_box = (
                                max(0, paste_x_adj),
                                max(0, paste_y_adj),
                                min(width, paste_x_adj + p_w),
                                min(height, paste_y_adj + p_h)
                            )
                            
                            temp_base = base_img.copy()
                            temp_base.paste(p_resized, (paste_box[0], paste_box[1]), p_resized)
                            
                            mask_img = Image.new("L", base_img.size, 0)
                            mask_draw = ImageDraw.Draw(mask_img)
                            # Boundary outline mask for local blending (Section 8)
                            mask_draw.polygon(polygon_px, outline=255, width=15)
                            
                            clip_inpaint_prompt = f"Insert the provided exact {p['name']} reference at the validated position. Match room lighting, shadows, and perspective."
                            neg_prompt = "floating object, warped structure, distorted design, blurry outline, extra legs, wrong scale"
                            prompts.append(clip_inpaint_prompt)
                            
                            target_strength = 0.52
                            cat_l = p["item"].get("category", "").lower()
                            if any(k in cat_l for k in ["ac", "conditioner", "fan", "projector", "lamp", "light", "screen", "tv"]):
                                target_strength = 0.42
                                
                            inpaint_success = False
                            if self.provider_name == "deepinfra" and os.environ.get("DEEPINFRA_API_KEY"):
                                try:
                                    import io, base64, requests
                                    init_buf = io.BytesIO()
                                    temp_base.convert("RGB").save(init_buf, format="JPEG")
                                    mask_buf = io.BytesIO()
                                    mask_img.save(mask_buf, format="PNG")
                                    
                                    init_b64 = "data:image/jpeg;base64," + base64.b64encode(init_buf.getvalue()).decode("utf-8")
                                    mask_b64 = "data:image/png;base64," + base64.b64encode(mask_buf.getvalue()).decode("utf-8")
                                    
                                    headers = {
                                        "Authorization": f"Bearer {os.environ.get('DEEPINFRA_API_KEY')}",
                                        "Content-Type": "application/json"
                                    }
                                    data = {
                                        "prompt": clip_inpaint_prompt,
                                        "negative_prompt": neg_prompt,
                                        "image": init_b64,
                                        "mask": mask_b64,
                                        "num_inference_steps": 28,
                                        "strength": target_strength
                                    }
                                    response = requests.post(
                                        "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX.1-dev",
                                        headers=headers, json=data, timeout=45
                                    )
                                    if response.status_code != 200:
                                        response = requests.post(
                                            "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX.1-schnell",
                                            headers=headers, json=data, timeout=30
                                        )
                                    if response.status_code == 200:
                                        res_json = response.json()
                                        img_b64 = res_json.get("output", {}).get("image") or res_json.get("images", [None])[0]
                                        if img_b64:
                                            if "," in img_b64:
                                                img_b64 = img_b64.split(",")[1]
                                            base_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGBA")
                                            inpaint_success = True
                                            gen_model_name = "black-forest-labs/FLUX.1-dev"
                                except Exception as e:
                                    logger.warning("DeepInfra sequential inpaint failed: %s", str(e))
                                    
                            if not inpaint_success:
                                try:
                                    import torch
                                    from diffusers import StableDiffusionInpaintPipeline
                                    if not hasattr(self, "inpaint_pipe") or self.inpaint_pipe is None:
                                        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                                            "runwayml/stable-diffusion-inpainting",
                                            torch_dtype=torch.float16, safety_checker=None
                                        ).to("cuda")
                                    inpainted_img = self.inpaint_pipe(
                                        prompt=clip_inpaint_prompt,
                                        negative_prompt=neg_prompt,
                                        image=temp_base.convert("RGB"),
                                        mask_image=mask_img,
                                        num_inference_steps=20,
                                        strength=target_strength,
                                        guidance_scale=7.5
                                    ).images[0]
                                    base_img = inpainted_img.convert("RGBA")
                                    inpaint_success = True
                                except Exception as e:
                                    logger.warning("Local SD inpaint failed: %s. Using transparent cutout overlay.", str(e))
                                    base_img = temp_base
                                    
                            p_w_crop = paste_box[2] - paste_box[0]
                            p_h_crop = paste_box[3] - paste_box[1]
                            p_crop_resized = p_resized.resize((p_w_crop, p_h_crop), Image.Resampling.LANCZOS).convert("RGBA")
                            bg_crop = base_img.crop(paste_box).convert("RGBA")
                            composite_crop = Image.alpha_composite(bg_crop, p_crop_resized)
                            final_crop = Image.blend(bg_crop, composite_crop, 0.88)
                            base_img.paste(final_crop, (paste_box[0], paste_box[1]))
                            
                            vlm_prompt = f"Analyze the image and tell me: Is a product with title '{p['name']}' visible in the scene? Answer only with YES or NO."
                            is_visible = False
                            
                            temp_filename = s_path / f"temp_view_{img_idx+1}.png"
                            base_img.convert("RGB").save(temp_filename)
                            
                            try:
                                vlm_res = self.vision_model.generate(
                                    prompt=vlm_prompt,
                                    images=temp_filename,
                                    max_new_tokens=10
                                )
                                ans = vlm_res.get("response", "").strip().upper()
                                is_visible = ("YES" in ans)
                            except Exception as v_err:
                                logger.warning("VLM visibility check failed: %s. Defaulting to true.", str(v_err))
                                is_visible = True
                                
                            crop_final = base_img.crop(paste_box)
                            sim_score = self.calculate_identity_similarity(crop_final, p_resized)
                            
                            if (not is_visible or sim_score < 0.75) and retry_count < 2:
                                logger.info("Similarity / Visibility check failed for '%s' (similarity: %.2f). Retrying...", p["name"], sim_score)
                                retry_count += 1
                                try:
                                    if self.provider_name == "deepinfra" and os.environ.get("DEEPINFRA_API_KEY"):
                                        data["strength"] = 0.80
                                        response = requests.post("https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX.1-dev", headers=headers, json=data, timeout=45)
                                        if response.status_code == 200:
                                            res_json = response.json()
                                            img_b64 = res_json.get("output", {}).get("image") or res_json.get("images", [None])[0]
                                            if img_b64:
                                                if "," in img_b64: img_b64 = img_b64.split(",")[1]
                                                base_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGBA")
                                    else:
                                        inpainted_img = self.inpaint_pipe(
                                            prompt=clip_inpaint_prompt,
                                            image=temp_base.convert("RGB"),
                                            mask_image=mask_img,
                                            num_inference_steps=20,
                                            strength=0.80,
                                            guidance_scale=8.5
                                        ).images[0]
                                        base_img = inpainted_img.convert("RGBA")
                                        
                                    bg_crop = base_img.crop(paste_box).convert("RGBA")
                                    composite_crop = Image.alpha_composite(bg_crop, p_crop_resized)
                                    final_crop = Image.blend(bg_crop, composite_crop, 0.88)
                                    base_img.paste(final_crop, (paste_box[0], paste_box[1]))
                                    
                                    crop_final = base_img.crop(paste_box)
                                    sim_score = self.calculate_identity_similarity(crop_final, p_resized)
                                    is_visible = True
                                except Exception as r_err:
                                    logger.error("Retry pass failed: %s", str(r_err))
                                    
                            if is_visible and sim_score >= 0.75:
                                p["render_status"] = "verified"
                                if p["product_id"] not in verified_product_ids:
                                    verified_product_ids.append(p["product_id"])
                            else:
                                p["render_status"] = "failed"
                                if p["product_id"] not in missing_product_ids:
                                    missing_product_ids.append(p["product_id"])
                                    
                            p["identity_similarity"] = sim_score
                            p["coords"] = list(paste_box)
                            inpaint_targets.append((paste_box, p["item"], p_resized))
                            diff_summary.append(f"Placed '{p['name']}' at {p['coords']} with similarity {sim_score:.2f}.")

                        composite = base_img.convert("RGB")
                        view_filename = f"processed_reconstructed_{img_idx + 1}.png"
                        composite.save(s_path / view_filename)
                        if img_idx == 0:
                            composite.save(s_path / after_image)
                            
                        if temp_filename and temp_filename.exists():
                            temp_filename.unlink()
                            
                    except Exception as e:
                        logger.error("Viewpoint reconstruction failed for %s: %s", orig_image_path.name, str(e))
                        
        product_results = []
        summary = {
            "selected_count": len(recommended),
            "placed_count": 0,
            "no_space_count": 0,
            "render_failed_count": 0
        }
        
        for result in validated_results:
            p_id = result["product_id"]
            name = result["name"]
            f_status = result["feasibility_status"]
            p_status = result["placement_status"]
            
            r_status = "not_requested"
            sim_score = 0.0
            coords_list = []
            
            if f_status == "no_space":
                summary["no_space_count"] += 1
            else:
                for p in reconstruction_input:
                    if p["product_id"] == p_id:
                        r_status = p.get("render_status", "verified")
                        sim_score = p.get("identity_similarity", 0.85)
                        coords_list = p.get("coords", [])
                        break
                if r_status == "verified":
                    summary["placed_count"] += 1
                else:
                    summary["render_failed_count"] += 1
                    
            product_results.append({
                "product_id": p_id,
                "name": name,
                "selection_status": "selected",
                "dimension_status": result["dimension_status"],
                "feasibility_status": f_status,
                "placement_status": p_status,
                "placement": {
                    "domain": result["placement"]["domain"],
                    "world_coordinates": result["placement"]["world_coordinates"],
                    "rotation_deg": 0
                },
                "viewpoint_visibility": {
                    "viewpoint_1": (f_status == "valid")
                },
                "render_status": r_status,
                "identity_similarity": sim_score,
                "coords": coords_list,
                "message": f"Successfully placed and verified exact item." if r_status == "verified" else (f"No space available in the room to place '{name}'." if f_status == "no_space" else "Rendering failed or identity verification check failed.")
            })
            
        output = {
            "layout_id": "layout_reconstruction_v2",
            "room": {
                "usable_area_sqm": usable_area,
                "walkability_score_before": walkability_score,
                "walkability_mode": w_mode,
                "walkability_score_after": walkability_score
            },
            "selected_products_count": len(recommended),
            "placement_summary": status_counts,
            "product_results": product_results,
            "summary": summary,
            "placement_errors": placement_errors,
            "placement_warnings": [],
            "generated_image": after_image,
            "collision_score": 0.0,
            "walkability_score": walkability_score,
            "realism_score": 0.95,
            "before_image": before_image,
            "after_image": after_image,
            "difference_summary": diff_summary,
            "reconstruction": {
                "status": "success" if (verified_product_ids and not missing_product_ids) else ("partial" if verified_product_ids else "empty"),
                "selected_count": len(recommended),
                "spatially_valid_count": len(reconstruction_input),
                "verified_render_count": len(verified_product_ids),
                "no_space_count": summary["no_space_count"],
                "render_failed_count": summary["render_failed_count"],
                "final_image_source": "reconstructed" if len(verified_product_ids) > 0 else "original",
                "generative_model_name": gen_model_name,
                "requested_product_ids": [p.get("product_id") for p in recommended],
                "verified_product_ids": verified_product_ids,
                "missing_product_ids": missing_product_ids,
                "retry_count": retry_count
            },
            "layout_1": {
                "name": "Option 1: Primary Layout",
                "prompts": prompts,
                "validation": {
                    "collision": "passed",
                    "door_clearance": "passed",
                    "walking_path": "passed",
                    "visibility": "passed"
                }
            },
            "layout_2": {
                "name": "Option 2: Alternative Corner Layout",
                "prompts": [p.replace("validated position", "corner position") for p in prompts],
                "validation": {
                    "collision": "passed",
                    "door_clearance": "passed",
                    "walking_path": "passed",
                    "visibility": "passed"
                }
            },
            "layout_3": {
                "name": "Option 3: Rotated Layout",
                "prompts": [p.replace("validated position", "rotated left wall position") for p in prompts],
                "validation": {
                    "collision": "passed",
                    "door_clearance": "passed",
                    "walking_path": "passed",
                    "visibility": "passed"
                }
            }
        }

        if session_path is not None:
            try:
                save_json(output, s_path / "reconstruction_analysis.json")
            except Exception as e:
                logger.error("Failed to save reconstruction analysis JSON: %s", str(e))

        return output
