# utils/room_mapper.py
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class RoomMapper:
    """Estimates room boundaries, projects object boxes to physical space,

    and generates spatial relationship graphs.
    """

    def map_room_coordinates(
        self, base_scene: Dict[str, Any], doors: List[Dict[str, Any]], windows: List[Dict[str, Any]]
    ) -> Tuple[int, int, int]:
        """Projects image-level bounding boxes into relative physical coordinates in centimeters.

        Origin (0, 0, 0) is the front-left floor corner.
        Returns:
            room_width, room_depth, room_height
        """
        # 1. Determine Room Bounds
        r_size = base_scene.get("room_understanding", {}).get("room_size", "medium").lower()
        if r_size == "large":
            room_width, room_depth, room_height = 500, 600, 260
        elif r_size == "small":
            room_width, room_depth, room_height = 300, 400, 240
        else:
            room_width, room_depth, room_height = 400, 500, 240

        # 2. Get image dimensions from metadata
        img_w, img_h = 800, 600
        meta = base_scene.get("image_metadata", [])
        if meta and isinstance(meta, list) and isinstance(meta[0], dict):
            img_w = meta[0].get("image_width") or 800
            img_h = meta[0].get("image_height") or 600

        # 3. Project objects
        for obj in base_scene.get("objects", []):
            bbox = obj.get("bbox", {"x": 100, "y": 100, "width": 200, "height": 200})
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2

            model_pos = obj.get("room_position")
            if isinstance(model_pos, dict) and model_pos.get("x_cm") is not None and model_pos.get("y_cm") is not None:
                x_cm = int(model_pos["x_cm"])
                y_cm = int(model_pos["y_cm"])
                rotation_deg = int(model_pos.get("rotation") or obj.get("rotation_deg") or 0)
            else:
                x_cm = int((cx / img_w) * room_width)
                y_cm = int((cy / img_h) * room_depth)

                orient = str(obj.get("orientation", "")).lower()
                if "vertical" in orient or "north" in orient or "south" in orient:
                    rotation_deg = 90
                elif "east" in orient or "west" in orient:
                    rotation_deg = 0
                else:
                    rotation_deg = 90 if "desk" in obj.get("name", "").lower() or "dresser" in obj.get("name", "").lower() else 0

            # Override with direct VLM rotation_deg if present
            model_rot = obj.get("rotation_deg")
            if model_rot is not None:
                try:
                    rotation_deg = int(model_rot)
                except ValueError:
                    pass

            name_cat = f"{obj.get('name', '')} {obj.get('category', '')}".lower()
            if "lamp" in name_cat or "sconce" in name_cat or "light" in name_cat:
                if "table" in name_cat or "desk" in name_cat:
                    z_cm = 75
                elif "ceiling" in name_cat:
                    z_cm = 220
                else:
                    z_cm = 120
            elif "painting" in name_cat or "frame" in name_cat or "decor" in name_cat:
                z_cm = 140
            else:
                z_cm = 0

            obj["position"] = {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "z_cm": z_cm
            }
            obj["room_position"] = {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "rotation": rotation_deg
            }
            obj["room_coordinates"] = [round(x_cm / 100.0, 2), round(y_cm / 100.0, 2), round(z_cm / 100.0, 2)]
            obj["rotation_deg"] = rotation_deg

        # 4. Project doors
        for door in doors:
            bbox = door.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 100})
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            
            model_pos = door.get("room_position")
            if isinstance(model_pos, dict) and model_pos.get("x_cm") is not None and model_pos.get("y_cm") is not None:
                x_cm = int(model_pos["x_cm"])
                y_cm = int(model_pos["y_cm"])
                rotation_deg = int(model_pos.get("rotation") or door.get("rotation_deg") or 0)
            else:
                x_cm = int((cx / img_w) * room_width)
                y_cm = int((cy / img_h) * room_depth)
                rotation_deg = int(door.get("rotation_deg") or 0)
            
            door["position"] = {"x_cm": x_cm, "y_cm": y_cm, "z_cm": 0}
            door["room_position"] = {"x_cm": x_cm, "y_cm": y_cm, "rotation": rotation_deg}
            door["room_coordinates"] = [round(x_cm / 100.0, 2), round(y_cm / 100.0, 2), 0.0]
            door["rotation_deg"] = rotation_deg

        # 5. Project windows
        for window in windows:
            bbox = window.get("bbox", {"x": 0, "y": 0, "width": 100, "height": 100})
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            
            model_pos = window.get("room_position")
            if isinstance(model_pos, dict) and model_pos.get("x_cm") is not None and model_pos.get("y_cm") is not None:
                x_cm = int(model_pos["x_cm"])
                y_cm = int(model_pos["y_cm"])
                rotation_deg = int(model_pos.get("rotation") or window.get("rotation_deg") or 0)
            else:
                x_cm = int((cx / img_w) * room_width)
                y_cm = int((cy / img_h) * room_depth)
                rotation_deg = int(window.get("rotation_deg") or 0)
            
            window["position"] = {"x_cm": x_cm, "y_cm": y_cm, "z_cm": 100}
            window["room_position"] = {"x_cm": x_cm, "y_cm": y_cm, "rotation": rotation_deg}
            window["room_coordinates"] = [round(x_cm / 100.0, 2), round(y_cm / 100.0, 2), 1.0]
            window["rotation_deg"] = rotation_deg

        return room_width, room_depth, room_height

    def map_single_scene_coordinates(
        self, scene: Dict[str, Any], room_width: int, room_depth: int, img_w: int = 800, img_h: int = 600
    ) -> None:
        """Projects room coordinates for a single scene prior to fusion."""
        for obj in scene.get("objects", []):
            bbox = obj.get("bbox", {"x": 100, "y": 100, "width": 200, "height": 200})
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            
            model_pos = obj.get("room_position")
            if isinstance(model_pos, dict) and model_pos.get("x_cm") is not None and model_pos.get("y_cm") is not None:
                x_cm = int(model_pos["x_cm"])
                y_cm = int(model_pos["y_cm"])
                rotation_deg = int(model_pos.get("rotation") or obj.get("rotation_deg") or 0)
            else:
                x_cm = int((cx / img_w) * room_width)
                y_cm = int((cy / img_h) * room_depth)
                rotation_deg = int(obj.get("rotation_deg") or 0)
                
            name_cat = f"{obj.get('name', '')} {obj.get('category', '')}".lower()
            if "lamp" in name_cat or "sconce" in name_cat or "light" in name_cat:
                z_cm = 75 if ("table" in name_cat or "desk" in name_cat) else 120
            else:
                z_cm = 0
                
            obj["position"] = {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "z_cm": z_cm
            }
            obj["room_position"] = {
                "x_cm": x_cm,
                "y_cm": y_cm,
                "rotation": rotation_deg
            }
            obj["room_coordinates"] = [round(x_cm / 100.0, 2), round(y_cm / 100.0, 2), round(z_cm / 100.0, 2)]
            obj["rotation_deg"] = rotation_deg

    def generate_spatial_relationships(
        self, objects: List[Dict[str, Any]], doors: List[Dict[str, Any]], windows: List[Dict[str, Any]], room_width: int, room_depth: int, img_w: int = 800
    ) -> List[Dict[str, Any]]:
        """Calculates distance matrices and adds geometric relationships."""
        relationships: List[Dict[str, Any]] = []

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                o1 = objects[i]
                o2 = objects[j]
                
                pos1 = o1.get("position", {"x_cm": 0, "y_cm": 0, "z_cm": 0})
                pos2 = o2.get("position", {"x_cm": 0, "y_cm": 0, "z_cm": 0})
                
                dx = pos1["x_cm"] - pos2["x_cm"]
                dy = pos1["y_cm"] - pos2["y_cm"]
                
                if dx > 80:
                    relationships.append({"object1": o1["id"], "object2": o2["id"], "relation": "right_of"})
                    relationships.append({"object1": o2["id"], "object2": o1["id"], "relation": "left_of"})
                elif dx < -80:
                    relationships.append({"object1": o1["id"], "object2": o2["id"], "relation": "left_of"})
                    relationships.append({"object1": o2["id"], "object2": o1["id"], "relation": "right_of"})
                    
                if dy > 80:
                    relationships.append({"object1": o1["id"], "object2": o2["id"], "relation": "behind"})
                    relationships.append({"object1": o2["id"], "object2": o1["id"], "relation": "in_front_of"})
                elif dy < -80:
                    relationships.append({"object1": o1["id"], "object2": o2["id"], "relation": "in_front_of"})
                    relationships.append({"object1": o2["id"], "object2": o1["id"], "relation": "behind"})

        # Alignments with walls
        for o in objects:
            pos = o.get("position", {"x_cm": 0, "y_cm": 0})
            if pos["x_cm"] < 80:
                relationships.append({"object1": o["id"], "object2": "wall_west", "relation": "against_wall"})
            if pos["x_cm"] > (room_width - 80):
                relationships.append({"object1": o["id"], "object2": "wall_east", "relation": "against_wall"})
            if pos["y_cm"] > (room_depth - 80):
                relationships.append({"object1": o["id"], "object2": "wall_north", "relation": "against_wall"})
            if pos["y_cm"] < 80:
                relationships.append({"object1": o["id"], "object2": "wall_south", "relation": "against_wall"})
                
            # Windows and Doors proximity
            for w in windows:
                w_pos_x = int((w.get("bbox", {}).get("x", 0) / img_w) * room_width)
                if abs(pos["x_cm"] - w_pos_x) < 150:
                    relationships.append({"object1": o["id"], "object2": w["id"], "relation": "near_window"})
            for d in doors:
                d_pos_x = int((d.get("bbox", {}).get("x", 0) / img_w) * room_width)
                if abs(pos["x_cm"] - d_pos_x) < 150:
                    relationships.append({"object1": o["id"], "object2": d["id"], "relation": "near_door"})

        return relationships
