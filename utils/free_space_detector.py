# utils/free_space_detector.py
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class FreeSpaceDetector:
    """Discretizes room layout into an occupancy grid and extracts free-space polygons."""

    def __init__(self, resolution_cm: int = 10) -> None:
        self.resolution = resolution_cm

    def build_occupancy_grid(
        self, room_width_cm: int, room_depth_cm: int, objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Creates a grid matrix where 0 is free space and 1 is occupied."""
        # Convert dimensions to grid cell coordinates
        grid_w = max(1, room_width_cm // self.resolution)
        grid_d = max(1, room_depth_cm // self.resolution)

        # Initialize empty grid (0 = free)
        cells = [[0 for _ in range(grid_w)] for _ in range(grid_d)]

        # Fill in occupied blocks from furniture dimensions
        for obj in objects:
            pos = obj.get("position", {"x_cm": 0, "y_cm": 0})
            dims = obj.get("estimated_dimensions") or {"width_cm": 80, "depth_cm": 80}
            
            w_cm = dims.get("width_cm") or 80.0
            d_cm = dims.get("depth_cm") or 80.0

            # Convert object bounds to cell coordinates
            px, py = pos["x_cm"], pos["y_cm"]
            
            # Map object relative to room bounds (clamp to valid ranges)
            min_x = max(0, int((px - w_cm/2) / self.resolution))
            max_x = min(grid_w, int((px + w_cm/2) / self.resolution))
            min_y = max(0, int((py - d_cm/2) / self.resolution))
            max_y = min(grid_d, int((py + d_cm/2) / self.resolution))

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    cells[y][x] = 1

        return {
            "grid_resolution_cm": self.resolution,
            "width": grid_w,
            "height": grid_d,
            "cells": cells
        }

    def detect_free_regions(
        self, occupancy_grid: Dict[str, Any], room_width_cm: int, room_depth_cm: int
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, int]]]]:
        """Identifies empty regions and compiles lists of free space polygons."""
        cells = occupancy_grid["cells"]
        grid_w = occupancy_grid["width"]
        grid_d = occupancy_grid["height"]

        # Divide grid into 4 sectors (quadrants) to extract clear regions
        occupied_sectors = set()
        for y in range(grid_d):
            for x in range(grid_w):
                if cells[y][x] == 1:
                    sector_x = "left" if x < (grid_w / 2) else "right"
                    sector_y = "front" if y < (grid_d / 2) else "back"
                    occupied_sectors.add((sector_x, sector_y))

        empty_regions: List[Dict[str, Any]] = []
        free_polygons: List[List[Dict[str, int]]] = []

        sector_coords = {
            ("left", "front"): {"x_range": (0, int(room_width_cm/2)), "y_range": (0, int(room_depth_cm/2))},
            ("right", "front"): {"x_range": (int(room_width_cm/2), room_width_cm), "y_range": (0, int(room_depth_cm/2))},
            ("left", "back"): {"x_range": (0, int(room_width_cm/2)), "y_range": (int(room_depth_cm/2), room_depth_cm)},
            ("right", "back"): {"x_range": (int(room_width_cm/2), room_width_cm), "y_range": (int(room_depth_cm/2), room_depth_cm)},
        }

        region_count = 1
        for sector, coords in sector_coords.items():
            if sector not in occupied_sectors:
                w_cm = coords["x_range"][1] - coords["x_range"][0]
                d_cm = coords["y_range"][1] - coords["y_range"][0]

                empty_regions.append({
                    "id": f"region_{region_count:03d}",
                    "width_cm": w_cm,
                    "depth_cm": d_cm,
                    "suitable_for": ["furniture", "seating", "storage"] if sector[1] == "front" else ["desk", "decor"]
                })

                # Format polygon coordinates as a list of points
                free_polygons.append([
                    {"x": coords["x_range"][0], "y": coords["y_range"][0]},
                    {"x": coords["x_range"][1], "y": coords["y_range"][0]},
                    {"x": coords["x_range"][1], "y": coords["y_range"][1]},
                    {"x": coords["x_range"][0], "y": coords["y_range"][1]}
                ])
                region_count += 1

        # Fallback if room is fully packed (or no grid quadrant is empty)
        if not empty_regions:
            w_cm = int(room_width_cm / 2)
            d_cm = 150
            empty_regions.append({
                "id": "region_001",
                "width_cm": w_cm,
                "depth_cm": d_cm,
                "suitable_for": ["seating", "decor"]
            })
            free_polygons.append([
                {"x": 0, "y": 0},
                {"x": w_cm, "y": 0},
                {"x": w_cm, "y": 150},
                {"x": 0, "y": 150}
            ])

        return empty_regions, free_polygons
