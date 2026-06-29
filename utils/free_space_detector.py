# utils/free_space_detector.py
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class FreeSpaceDetector:
    """Discretizes room layout into an occupancy grid and extracts free-space polygons."""

    def __init__(self, resolution_cm: int = 5) -> None:
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
        """Identifies empty regions and compiles lists of free space polygons using BFS connected components."""
        cells = occupancy_grid["cells"]
        grid_w = occupancy_grid["width"]
        grid_d = occupancy_grid["height"]

        visited = [[False for _ in range(grid_w)] for _ in range(grid_d)]
        empty_regions: List[Dict[str, Any]] = []
        free_polygons: List[List[Dict[str, int]]] = []
        region_count = 1

        for y in range(grid_d):
            for x in range(grid_w):
                if cells[y][x] == 0 and not visited[y][x]:
                    # Start BFS to find the connected components
                    component = []
                    queue = [(x, y)]
                    visited[y][x] = True

                    while queue:
                        cx, cy = queue.pop(0)
                        component.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < grid_w and 0 <= ny < grid_d:
                                if cells[ny][nx] == 0 and not visited[ny][nx]:
                                    visited[ny][nx] = True
                                    queue.append((nx, ny))

                    # Calculate bounding box of the component
                    min_x = min(pt[0] for pt in component)
                    max_x = max(pt[0] for pt in component)
                    min_y = min(pt[1] for pt in component)
                    max_y = max(pt[1] for pt in component)

                    w_cm = (max_x - min_x + 1) * self.resolution
                    d_cm = (max_y - min_y + 1) * self.resolution

                    # Include only components larger than 50cm x 50cm
                    if w_cm >= 50 and d_cm >= 50:
                        empty_regions.append({
                            "id": f"region_{region_count:03d}",
                            "width_cm": w_cm,
                            "depth_cm": d_cm,
                            "suitable_for": ["furniture", "seating", "storage"] if w_cm >= 100 else ["desk", "decor"]
                        })

                        # Bounding box coordinates
                        free_polygons.append([
                            {"x": min_x * self.resolution, "y": min_y * self.resolution},
                            {"x": (max_x + 1) * self.resolution, "y": min_y * self.resolution},
                            {"x": (max_x + 1) * self.resolution, "y": (max_y + 1) * self.resolution},
                            {"x": min_x * self.resolution, "y": (max_y + 1) * self.resolution}
                        ])
                        region_count += 1

        # Fallback if no empty region found
        if not empty_regions:
            w_cm = min(room_width_cm, 150)
            d_cm = min(room_depth_cm, 150)
            empty_regions.append({
                "id": "region_001",
                "width_cm": w_cm,
                "depth_cm": d_cm,
                "suitable_for": ["seating", "decor"]
            })
            free_polygons.append([
                {"x": 0, "y": 0},
                {"x": w_cm, "y": 0},
                {"x": w_cm, "y": d_cm},
                {"x": 0, "y": d_cm}
            ])

        return empty_regions, free_polygons
