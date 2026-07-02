"""Vision Agent Module for VisionSpace AI.

Orchestrates input ingestion, image validation, configurable preprocessing,
session logging, LLM inference via ModelManager/VisionModel facades, JSON response validation/repair,
and multi-view scene merge/fusion.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config
from models.vision_model import VisionModel
from models.model_manager import ModelManager
from utils.vision_merger import VisionMerger, _normalise_ids_list
from utils.object_fusion import ObjectFusion
from utils.structural_merger import StructuralMerger
from utils.room_mapper import RoomMapper
from utils.free_space_detector import FreeSpaceDetector
from utils.prompt_templates import (
    vision_analysis_prompt,
    vision_tile_analysis_prompt,
    vision_crop_analysis_prompt
)
from utils.file_utils import save_json, ensure_output_dirs

logger = logging.getLogger("VisionSpaceAI.VisionAgent")


class VisionAgent:
    """Production-ready VisionAgent responsible for complete room scene understanding orchestration."""

    def __init__(
        self,
        model_profile: str = "Balanced",
        provider: str = "huggingface",
        output_dir: Optional[str] = None,
    ) -> None:
        """Initializes the VisionAgent.

        Args:
            model_profile: Model profile setting ("Fast", "Balanced", "Accurate").
            provider: Inference execution provider ("huggingface").
            output_dir: Target output root directory (defaults to config.OUTPUT_DIR).
        """
        self.model_profile = model_profile
        self.provider_name = provider
        self.output_dir = output_dir or getattr(config, "OUTPUT_DIR", "outputs")
        
        # Facade VisionModel reference
        self.vision_model = VisionModel(provider=provider)
        self.model_manager = ModelManager()

        logger.info(
            "VisionAgent initialized (Profile: %s, Provider: %s, Outputs: %s)",
            model_profile, provider, self.output_dir
        )

    # -------------------------------------------------------------------------
    # 1. Input Normalization
    # -------------------------------------------------------------------------
    def normalize_input(self, source: Any) -> List[Image.Image]:
        """Converts diverse image formats into standard RGB PIL Image structures.

        Supports single/multiple paths, folders, PIL images, NumPy arrays, OpenCV formats,
        image bytes, Base64 strings, and URL downloads.
        """
        if source is None:
            return []

        # If it's a list or tuple, process recursively
        if isinstance(source, (list, tuple)):
            results = []
            for item in source:
                results.extend(self.normalize_input(item))
            return results

        # 1. Path-like strings / pathlib.Path
        if isinstance(source, (str, pathlib.Path)):
            # Check base64 pattern first before path checks
            if isinstance(source, str) and (source.startswith("data:image") or len(source) > 512):
                return self._parse_base64(source)

            path = pathlib.Path(source)
            if path.exists():
                if path.is_dir():
                    # Scan directory
                    images = []
                    supported_exts = getattr(config, "SUPPORTED_IMAGE_TYPES", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
                    for ext in supported_exts:
                        images.extend(path.glob(f"*{ext}"))
                        images.extend(path.glob(f"*{ext.upper()}"))
                    
                    images = sorted(list(set(images)))
                    logger.info("Found %d image files in directory %s", len(images), path)
                    return [self._load_image_file(p) for p in images]
                else:
                    return [self._load_image_file(path)]
            elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                return [self._download_url(source)]
            else:
                raise FileNotFoundError(f"Source file or path does not exist: {source}")

        # 2. PIL Image
        if isinstance(source, Image.Image):
            return [source]

        # 3. NumPy / OpenCV Array
        if isinstance(source, np.ndarray):
            return [self._parse_numpy(source)]

        # 4. Raw Bytes
        if isinstance(source, bytes):
            return [Image.open(io.BytesIO(source))]

        raise TypeError(f"Unsupported input source type: {type(source)}")

    def _load_image_file(self, path: pathlib.Path) -> Image.Image:
        try:
            return Image.open(path)
        except Exception as e:
            logger.error("Failed to load local file image %s: %s", path, str(e))
            raise

    def _download_url(self, url: str) -> Image.Image:
        try:
            response = requests.get(url, timeout=30.0)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.error("Failed to download image URL %s: %s", url, str(e))
            raise

    def _parse_base64(self, b64_str: str) -> List[Image.Image]:
        try:
            # Strip data URI header if present
            clean_b64 = re.sub(r'^data:image/.+;base64,', '', b64_str)
            decoded = base64.b64decode(clean_b64)
            return [Image.open(io.BytesIO(decoded))]
        except Exception as e:
            logger.error("Base64 image decoding failed: %s", str(e))
            raise ValueError(f"Could not parse Base64 string: {str(e)}")

    def _parse_numpy(self, arr: np.ndarray) -> Image.Image:
        try:
            # If shape is 3-dimensional (Height, Width, Channels)
            if len(arr.shape) == 3:
                # Standard OpenCV formats are BGR; convert BGR -> RGB via NumPy slicing
                if arr.shape[2] == 3:
                    arr_rgb = arr[:, :, ::-1]
                    return Image.fromarray(arr_rgb)
                elif arr.shape[2] == 4:
                    # BGRA -> RGBA conversion
                    arr_rgba = arr.copy()
                    arr_rgba[:, :, 0], arr_rgba[:, :, 2] = arr[:, :, 2], arr[:, :, 0]
                    return Image.fromarray(arr_rgba)
            return Image.fromarray(arr)
        except Exception as e:
            logger.error("NumPy array conversion to PIL failed: %s", str(e))
            raise ValueError(f"Could not parse NumPy array: {str(e)}")

    # -------------------------------------------------------------------------
    # 2. Image Validation & Deduplication
    # -------------------------------------------------------------------------
    def validate_and_deduplicate(self, images: List[Image.Image]) -> List[Image.Image]:
        """Validates image integrity and deduplicates identical images via pixel hashes."""
        valid_images = []
        seen_hashes = set()

        for idx, img in enumerate(images):
            if img is None:
                continue

            # Verify integrity
            try:
                img.load()
            except Exception as e:
                logger.warning("Corrupt image detected at index %d, skipping: %s", idx, str(e))
                continue

            # Check positive dimensions
            if img.width <= 0 or img.height <= 0:
                logger.warning("Invalid dimensions %dx%d at index %d, skipping.", img.width, img.height, idx)
                continue

            # Alpha conversion to RGB
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Check content deduplication via MD5 hash of pixel data
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                valid_images.append(img)
            else:
                logger.info("Skipping duplicate image at index %d based on pixel hashing", idx)

        # Enforce max limits
        max_imgs = getattr(config, "MAX_IMAGES_PER_REQUEST", 6)
        if len(valid_images) > max_imgs:
            logger.warning("Image count %d exceeds limit (%d). Capping list.", len(valid_images), max_imgs)
            valid_images = valid_images[:max_imgs]

        return valid_images

    # -------------------------------------------------------------------------
    # 3. Configurable Preprocessing Pipeline
    # -------------------------------------------------------------------------
    def preprocess_image(
        self,
        img: Image.Image,
        resize_dim: int = 1024,
        noise_reduction: bool = False,
        enhance_contrast: float = 1.0,
        enhance_brightness: float = 1.0,
        normalize_pixels: bool = False,
    ) -> Image.Image:
        """Executes configurable image preparation steps."""
        # 1. EXIF orientation correction
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            logger.debug("EXIF transposition bypassed: %s", str(e))

        # 2. Resize keeping aspect ratio
        width, height = img.size
        max_side = max(width, height)
        if max_side > resize_dim:
            ratio = resize_dim / max_side
            new_w = int(width * ratio)
            new_h = int(height * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 3. Optional Noise Reduction
        if noise_reduction:
            img = img.filter(ImageFilter.MedianFilter(size=3))

        # 4. Optional Enhancements
        if enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(enhance_contrast)

        if enhance_brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(enhance_brightness)

        # 5. Optional normalization check (handled internally by processor, but could map here if desired)
        if normalize_pixels:
            # Placeholder for custom contrast normalization if requested
            img = ImageOps.autocontrast(img)

        return img

    # -------------------------------------------------------------------------
    # 4. Orchestrate Scene Analysis
    # -------------------------------------------------------------------------
    def analyze_room(
        self,
        images: Union[Any, List[Any]],
        room_type: str = "living_room",
        style_preference: str = "modern",
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Main room scene understanding workflow (Synchronous)."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        t_total_start = time.perf_counter()
        
        # Setup session dir structure
        session_path = pathlib.Path(self.output_dir) / session_id
        orig_dir = session_path / "original_images"
        proc_dir = session_path / "processed_images"
        orig_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("Loading Images...")
        
        t_load_start = time.perf_counter()
        raw_images = self.normalize_input(images)
        valid_images = self.validate_and_deduplicate(raw_images)
        t_load_duration = time.perf_counter() - t_load_start

        if not valid_images:
            raise ValueError("No valid images provided for scene analysis.")

        if progress_callback:
            progress_callback("Preprocessing...")

        t_preprocess_start = time.perf_counter()
        preprocessed_images = []
        for idx, img in enumerate(valid_images):
            # Save original
            orig_name = f"original_{idx + 1}.png"
            img.save(orig_dir / orig_name)

            # Apply configurable pipeline
            proc_img = self.preprocess_image(
                img,
                resize_dim=kwargs.get("resize_dim", 1024),
                noise_reduction=kwargs.get("noise_reduction", False),
                enhance_contrast=kwargs.get("enhance_contrast", 1.0),
                enhance_brightness=kwargs.get("enhance_brightness", 1.0),
                normalize_pixels=kwargs.get("normalize_pixels", False),
            )
            preprocessed_images.append(proc_img)

            # Save processed
            proc_name = f"processed_{idx + 1}.png"
            proc_img.save(proc_dir / proc_name)

        t_preprocess_duration = time.perf_counter() - t_preprocess_start

        # Inference loop
        single_outputs = []
        t_inference_duration = 0.0

        for idx, p_img in enumerate(preprocessed_images):
            if progress_callback:
                progress_callback(f"Running Vision Model (Image {idx + 1}/{len(preprocessed_images)})...")

            # Dynamic prompt injection context details
            p_template = vision_analysis_prompt(
                room_type=room_type,
                confidence_threshold=kwargs.get("confidence_threshold", getattr(config, "VISION_CONFIG", {}).get("min_confidence", 0.35)),
                max_objects=kwargs.get("max_objects", 30)
            )

            # Insert runtime metadata instructions directly to guide model
            mv_instructions = ""
            if len(preprocessed_images) > 1:
                mv_instructions = (
                    f"\nThis is image index {idx + 1} of a multi-view series containing {len(preprocessed_images)} views. "
                    "Focus on identifying spatial details, camera angles, and objects relative to other views."
                )
            user_prompt = p_template["user"] + mv_instructions

            t_inf_start = time.perf_counter()
            response_data = self._generate_with_retry(
                prompt=user_prompt,
                image=p_img,
                system_prompt=p_template["system"],
                **kwargs
            )
            t_inference_duration += time.perf_counter() - t_inf_start

            if progress_callback:
                progress_callback(f"Parsing JSON (Image {idx + 1}/{len(preprocessed_images)})...")

            # Guard: raise early if inference failed (provider returns success=False)
            if not response_data.get("success", True) or not response_data.get("response", "").strip():
                provider_error = response_data.get("error", "Unknown inference error")
                raise RuntimeError(
                    f"VisionModel inference failed for image {idx + 1}: {provider_error}"
                )

            # Validate schema format
            parsed_json = self._validate_and_repair_json(response_data.get("response", ""))
            
            # Run image tiling for high-resolution detail extraction
            enable_tiling = kwargs.get("enable_tiling", config.VISION_CONFIG.get("enable_tiling", True))
            if enable_tiling:
                # Proactively reclaim VRAM before starting memory-intensive tiling pass
                self._clear_memory()
                if progress_callback:
                    progress_callback(f"Running Tile-Inference Pass (Image {idx + 1}/{len(preprocessed_images)})...")
                
                # Split image into 2x2 grid tiles
                tiles = self._split_image_into_tiles(p_img)
                tile_objects = []
                
                for t_idx, (tile_img, t_box) in enumerate(tiles):
                    t_template = vision_tile_analysis_prompt(
                        room_type=room_type,
                        tile_index=t_idx + 1,
                        confidence_threshold=kwargs.get("confidence_threshold", getattr(config, "VISION_CONFIG", {}).get("min_confidence", 0.35)),
                    )
                    # Allocate enough tokens to fit CoT thoughts and full JSON payload
                    tile_kwargs = dict(kwargs)
                    tile_kwargs["max_new_tokens"] = 1024
                    
                    # Append strict JSON-only instruction for thinking models
                    tile_prompt = t_template["user"] + "\n\nIMPORTANT: Do not output any conversational introduction or explanation outside the <think>...</think> block. Immediately output the final raw JSON block after the thoughts."
                    
                    t_response = self._generate_with_retry(
                        prompt=tile_prompt,
                        image=tile_img,
                        system_prompt=t_template["system"],
                        **tile_kwargs
                    )
                    
                    if t_response.get("success", True) and t_response.get("response", "").strip():
                        try:
                            t_json = self._validate_and_repair_json(t_response.get("response", ""))
                            for obj in t_json.get("objects", []):
                                if "bbox" in obj:
                                    # Map coordinates back to global space
                                    obj["bbox"] = self._map_local_bbox_to_global(obj["bbox"], t_box, tile_img.size)
                                    # Ensure unique ID namespace
                                    obj["id"] = f"{obj.get('id', 'item')}_t{t_idx + 1}"
                                    tile_objects.append(obj)
                        except Exception as e:
                            logger.warning(f"Error parsing tile {t_idx + 1}: {str(e)}")
                
                if tile_objects:
                    parsed_json["objects"].extend(tile_objects)

            # Run adaptive crop pass for focused furniture items
            enable_adaptive_crops = kwargs.get("enable_adaptive_crops", config.VISION_CONFIG.get("enable_adaptive_crops", True))
            if enable_adaptive_crops:
                # Proactively reclaim VRAM before starting memory-intensive crop pass
                self._clear_memory()
                # Find target furniture categories to crop
                target_categories = {"desk", "table", "bed", "sofa", "nightstand", "dresser", "shelf", "cabinet", "wardrobe", "tv", "television", "entertainment"}
                targets = []
                for obj in parsed_json.get("objects", []):
                    cat = str(obj.get("category", "")).lower()
                    subcat = str(obj.get("subcategory", "")).lower()
                    name = str(obj.get("name", "")).lower()
                    if cat in target_categories or subcat in target_categories or any(t in name for t in target_categories):
                        if "bbox" in obj:
                            targets.append(obj)
                
                # Limit to top 4 largest target furniture objects to scan all surfaces
                def get_bbox_area(obj):
                    bbox = obj.get("bbox", {})
                    return float(bbox.get("width", 0)) * float(bbox.get("height", 0))
                
                targets.sort(key=get_bbox_area, reverse=True)
                max_crops = kwargs.get("max_crops", getattr(config, "VISION_CONFIG", {}).get("max_crops", 2))
                targets_to_crop = targets[:max_crops]
                
                crop_objects = []
                for crop_idx, target in enumerate(targets_to_crop):
                    if progress_callback:
                        progress_callback(f"Running Crop-Inference Pass for {target.get('name')} (Image {idx + 1}/{len(preprocessed_images)})...")
                    
                    bbox = target["bbox"]
                    x, y, w_val, h_val = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
                    img_w, img_h = p_img.size
                    
                    # Expand crop window by 20% margin to capture surrounding objects
                    margin_w = w_val * 0.2
                    margin_h = h_val * 0.2
                    
                    left = max(0, int(x - margin_w))
                    top = max(0, int(y - margin_h))
                    right = min(img_w, int(x + w_val + margin_w))
                    bottom = min(img_h, int(y + h_val + margin_h))
                    
                    # Prevent empty crops
                    if right > left and bottom > top:
                        crop_box = (left, top, right, bottom)
                        crop_img = p_img.crop(crop_box)
                        
                        c_template = vision_crop_analysis_prompt(
                            furniture_name=target.get("name", "furniture"),
                            room_type=room_type,
                            confidence_threshold=kwargs.get("confidence_threshold", getattr(config, "VISION_CONFIG", {}).get("min_confidence", 0.35)),
                        )
                        crop_kwargs = dict(kwargs)
                        crop_kwargs["max_new_tokens"] = 1024
                        
                        # Append strict JSON-only instruction for thinking models
                        crop_prompt = c_template["user"] + "\n\nIMPORTANT: Do not output any conversational introduction or explanation outside the <think>...</think> block. Immediately output the final raw JSON block after the thoughts."
                        
                        c_response = self._generate_with_retry(
                            prompt=crop_prompt,
                            image=crop_img,
                            system_prompt=c_template["system"],
                            **crop_kwargs
                        )
                        
                        if c_response.get("success", True) and c_response.get("response", "").strip():
                            try:
                                c_json = self._validate_and_repair_json(c_response.get("response", ""))
                                for obj in c_json.get("objects", []):
                                    if "bbox" in obj:
                                        # Map coordinates back to global space
                                        obj["bbox"] = self._map_local_bbox_to_global(obj["bbox"], crop_box, crop_img.size)
                                        # Ensure unique ID namespace
                                        obj["id"] = f"{obj.get('id', 'item')}_c{crop_idx + 1}"
                                        crop_objects.append(obj)
                            except Exception as e:
                                logger.warning(f"Error parsing crop pass for {target.get('name')}: {str(e)}")
                
                if crop_objects:
                    parsed_json["objects"].extend(crop_objects)

            # Run local Object Fusion per viewpoint to clean up duplicate detections between tiles/crops
            if parsed_json.get("objects"):
                fusion = ObjectFusion()
                fused_viewpoint_objects = fusion.fuse_objects(parsed_json["objects"])
                parsed_json["objects"] = _normalise_ids_list(fused_viewpoint_objects)

            # Save single image intermediate output
            save_json(parsed_json, session_path / f"image_{idx + 1}_output.json")
            single_outputs.append(parsed_json)

        # Merge outputs if multi-image
        if progress_callback:
            progress_callback("Merging...")

        t_merge_start = time.perf_counter()
        if len(single_outputs) > 1:
            merged_output = self._merge_multi_view(single_outputs, session_path)
        else:
            merged_output = self._merge_multi_view([single_outputs[0]], session_path)

        # Add visual schema helper tags for downstream visual agents
        merged_output["segmentation_masks"] = []
        merged_output["placement_regions"] = merged_output.get("candidate_furniture_locations", [])
        
        # Populate free space polygon schemas from floor analyses (fallback only)
        if "free_space_polygons" not in merged_output or not merged_output["free_space_polygons"]:
            merged_output["free_space_polygons"] = merged_output.get("walking_paths", [])

        t_merge_duration = time.perf_counter() - t_merge_start

        if progress_callback:
            progress_callback("Saving...")

        # Save merged scene graph (remove raw 2D occupancy grid cells to keep JSON clean)
        import copy
        merged_output_copy = copy.deepcopy(merged_output)
        if "occupancy_grid" in merged_output_copy and isinstance(merged_output_copy["occupancy_grid"], dict):
            merged_output_copy["occupancy_grid"].pop("cells", None)
            merged_output_copy["occupancy_grid"]["occupancy_grid_path"] = "occupancy_grid.json"
        save_json(merged_output_copy, session_path / "merged_output.json")

        t_total_duration = time.perf_counter() - t_total_start

        # VRAM and hardware status info
        vram_info = self.model_manager.get_memory_usage()

        # Session metadata
        metadata = {
            "session_id": session_id,
            "session_path": str(session_path),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_name": getattr(config, "MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            "prompt_version": "25-feature-scene-understanding-v1",
            "image_count": len(preprocessed_images),
            "gpu_device": vram_info.get("device_name", "N/A") if torch.cuda.is_available() else "CPU",
            "performance_metrics": {
                "load_time_s": round(t_load_duration, 4),
                "preprocess_time_s": round(t_preprocess_duration, 4),
                "inference_time_s": round(t_inference_duration, 4),
                "merge_time_s": round(t_merge_duration, 4),
                "total_time_s": round(t_total_duration, 4)
            }
        }
        save_json(metadata, session_path / "metadata.json")

        # Session logs
        logs = {
            "session_id": session_id,
            "logs": [
                f"[{datetime.now().isoformat()}] Session started.",
                f"[{datetime.now().isoformat()}] Loaded {len(raw_images)} inputs, normalized to {len(valid_images)} unique images.",
                f"[{datetime.now().isoformat()}] Completed image preprocessing pipeline.",
                f"[{datetime.now().isoformat()}] Finished LLM inference runs.",
                f"[{datetime.now().isoformat()}] Merged multi-view inputs.",
                f"[{datetime.now().isoformat()}] Session files successfully saved."
            ]
        }
        save_json(logs, session_path / "logs.json")

        # Cleanup memory
        self._clear_memory()

        return {
            "session_id": session_id,
            "metadata": metadata,
            "merged_scene": merged_output,
            "individual_scenes": single_outputs
        }

    # -------------------------------------------------------------------------
    # 5. Async Scene Analysis Wrapper
    # -------------------------------------------------------------------------
    async def analyze_room_async(
        self,
        images: Union[Any, List[Any]],
        room_type: str = "living_room",
        style_preference: str = "modern",
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Runs room scene understanding workflow asynchronously (Non-blocking)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze_room(
                images=images,
                room_type=room_type,
                style_preference=style_preference,
                progress_callback=progress_callback,
                **kwargs
            )
        )

    # -------------------------------------------------------------------------
    # 6. Batch Scene Analysis
    # -------------------------------------------------------------------------
    @classmethod
    def analyze_batch(
        cls,
        batch_tasks: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Processes a batch of multiple room tasks sequentially."""
        agent = cls(
            model_profile=kwargs.get("model_profile", "Balanced"),
            provider=kwargs.get("provider", "huggingface"),
            output_dir=kwargs.get("output_dir")
        )
        
        batch_results = []
        for idx, task in enumerate(batch_tasks):
            logger.info("Processing batch task %d/%d", idx + 1, len(batch_tasks))
            try:
                res = agent.analyze_room(
                    images=task["images"],
                    room_type=task.get("room_type", "living_room"),
                    style_preference=task.get("style_preference", "modern"),
                    **kwargs
                )
                batch_results.append({"status": "success", "task_index": idx, "result": res})
            except Exception as e:
                logger.error("Batch task %d failed: %s", idx, str(e))
                batch_results.append({"status": "failed", "task_index": idx, "error": str(e)})
        return batch_results

    def _split_image_into_tiles(self, img: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Splits an image into a 2x2 grid of tiles.
        
        Returns a list of tuples: (tile_image, (left, top, right, bottom) relative to original image size).
        """
        w, h = img.size
        half_w = w // 2
        half_h = h // 2
        
        # Define crop coordinates
        boxes = [
            (0, 0, half_w, half_h),          # Top-Left
            (half_w, 0, w, half_h),          # Top-Right
            (0, half_h, half_w, h),          # Bottom-Left
            (half_w, half_h, w, h),          # Bottom-Right
        ]
        
        tiles = []
        for box in boxes:
            tiles.append((img.crop(box), box))
        return tiles

    def _map_local_bbox_to_global(
        self,
        bbox: Dict[str, float],
        crop_box: Tuple[int, int, int, int],
        source_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """Maps bounding box coordinates from a crop/tile back to global image coordinates.
        
        Supports both 0-1000 normalized coordinates and raw pixel coordinates.
        """
        left, top, right, bottom = crop_box
        crop_w = right - left
        crop_h = bottom - top
        src_w, src_h = source_size
        
        local_x = float(bbox.get("x", 0))
        local_y = float(bbox.get("y", 0))
        local_w = float(bbox.get("width", 0))
        local_h = float(bbox.get("height", 0))
        
        # Check if coordinates are normalized (0-1000 range)
        is_normalized_1000 = False
        if local_x <= 1000 and local_y <= 1000 and local_w <= 1000 and local_h <= 1000:
            if crop_w != 1000 and crop_h != 1000:
                is_normalized_1000 = True
                
        if is_normalized_1000:
            global_x = left + (local_x / 1000.0) * crop_w
            global_y = top + (local_y / 1000.0) * crop_h
            global_w = (local_w / 1000.0) * crop_w
            global_h = (local_h / 1000.0) * crop_h
        else:
            # Scale coordinates relative to source_size passed to the model
            global_x = left + (local_x / src_w) * crop_w
            global_y = top + (local_y / src_h) * crop_h
            global_w = (local_w / src_w) * crop_w
            global_h = (local_h / src_h) * crop_h
            
        return {
            "x": round(global_x, 1),
            "y": round(global_y, 1),
            "width": round(global_w, 1),
            "height": round(global_h, 1)
        }

    # -------------------------------------------------------------------------
    # Helper: LLM Inference with Retry Recovery
    # -------------------------------------------------------------------------
    def _generate_with_retry(
        self,
        prompt: str,
        image: Image.Image,
        system_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs image token generation, catching CUDA OOMs and executing dynamic reloads."""
        # Filter kwargs to only those accepted by VisionModel.generate
        allowed_keys = {
            "max_new_tokens", "temperature", "top_p", "do_sample", "stream", "profile", "response_format"
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        if "profile" not in filtered_kwargs and self.model_profile:
            filtered_kwargs["profile"] = self.model_profile

        try:
            # Route to VisionModel facade
            return self.vision_model.generate(
                prompt=prompt,
                images=[image],
                system_prompt=system_prompt,
                response_format="json",
                **filtered_kwargs
            )
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower() or isinstance(e, RuntimeError):
                logger.warning("CUDA OOM or error during Vision Agent generation. Reclaiming memory & reloading.")
                self._clear_memory()
                self.vision_model.reload()
                
                # Retry once
                return self.vision_model.generate(
                    prompt=prompt,
                    images=[image],
                    system_prompt=system_prompt,
                    response_format="json",
                    **filtered_kwargs
                )
            else:
                raise e

    # -------------------------------------------------------------------------
    # Helper: JSON Validation & Repair
    # -------------------------------------------------------------------------
    def _validate_and_repair_json(self, response_text: str) -> Dict[str, Any]:
        """Cleans response formatting and populates missing fields with null values."""
        clean_text = response_text.strip()
        
        # Strip thinking block (<think>...</think>) if present
        if "<think>" in clean_text.lower() and "</think>" in clean_text.lower():
            idx = clean_text.lower().find("</think>")
            clean_text = clean_text[idx + len("</think>"):].strip()
        
        # Strip markdown frames if present
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if len(lines) > 1:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()
            else:
                clean_text = clean_text.split("```")[1]
        
        if clean_text.lower().startswith("json"):
            clean_text = clean_text[4:].strip()
            
        clean_text = clean_text.strip()

        # Try multiple parsing and repair strategies
        data = None
        
        # Strategy 1: Direct parse
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove trailing commas and replace empty values after colons
        if data is None:
            try:
                repaired = re.sub(r',\s*([\]}])', r'\1', clean_text)
                # Replace empty values after colons, e.g. ": ," or ": }" with ": null"
                repaired = re.sub(r':\s*(,)', r': null\1', repaired)
                repaired = re.sub(r':\s*(\})', r': null\1', repaired)
                data = json.loads(repaired)
            except Exception:
                pass

        # Strategy 3: Truncation boundary recovery
        if data is None:
            candidates = ["},", "}", "],", "]"]
            for cand in candidates:
                idx = clean_text.rfind(cand)
                if idx == -1:
                    continue
                try:
                    prefix = clean_text[:idx + len(cand)].strip()
                    if prefix.endswith(","):
                        prefix = prefix[:-1].strip()
                        
                    stack = []
                    in_string = False
                    escape = False
                    for char in prefix:
                        if char == '"' and not escape:
                            in_string = not in_string
                        elif not in_string:
                            if char == "{":
                                stack.append("}")
                            elif char == "[":
                                stack.append("]")
                            elif char == "}" and stack and stack[-1] == "}":
                                stack.pop()
                            elif char == "]" and stack and stack[-1] == "]":
                                stack.pop()
                        if char == "\\" and not escape:
                            escape = True
                        else:
                            escape = False
                            
                    repaired_balanced = prefix
                    while stack:
                        repaired_balanced += stack.pop()
                        
                    data = json.loads(repaired_balanced)
                    if data is not None:
                        break
                except Exception:
                    pass

        # Strategy 4: Fallback to json-repair library if installed
        if data is None:
            try:
                from json_repair import repair_json
                repaired_lib = repair_json(clean_text, return_objects=True)
                if isinstance(repaired_lib, dict):
                    data = repaired_lib
            except Exception:
                pass

        # If all repairs failed, log the raw response and run direct parse to raise the original JSONDecodeError for debugging
        if data is None:
            logger.error("JSON repair failed. Raw response text:\n%s", clean_text)
            print("--> [DEBUG] Failed Raw JSON Response from Model:\n", clean_text)
            data = json.loads(clean_text)

        # Conform output payload structures
        schema_defaults = {
            "room_understanding": {},
            "image_metadata": [],
            "scene_coverage_percentage": None,
            "floor_analysis": {},
            "wall_analysis": {},
            "ceiling_analysis": {},
            "objects": [],
            "spatial_relationships": [],
            "doors": [],
            "windows": [],
            "lighting_analysis": {},
            "free_space_analysis": {},
            "accessibility": {},
            "walking_paths": [],
            "layout_problems": [],
            "empty_regions": [],
            "candidate_furniture_locations": [],
            "style": None,
            "dominant_color_palette": [],
            "overall_aesthetic": None,
            "image_quality": [],
            "safety_analysis": []
        }

        # Populate missing structures with defaults/nulls
        for key, val in schema_defaults.items():
            if key not in data:
                data[key] = val
            elif isinstance(val, dict) and isinstance(data[key], dict):
                # Ensure child dictionary keys conform
                pass # Can expand nested validation here if needed

        return data

    # -------------------------------------------------------------------------
    # Helper: Multi-view Merge / Scene Fusion
    # -------------------------------------------------------------------------
    def _merge_multi_view(self, scenes: List[Dict[str, Any]], session_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
        """Combines multiple single-image scene graphs into one canonical room scene graph."""
        # 1. Project physical room coordinates for each individual viewpoint scene's objects first
        r_size = scenes[0].get("room_understanding", {}).get("room_size", "medium").lower()
        if r_size == "large":
            room_width, room_depth = 500, 600
        elif r_size == "small":
            room_width, room_depth = 300, 400
        else:
            room_width, room_depth = 400, 500

        mapper = RoomMapper()
        for scene in scenes:
            img_w = 800
            meta = scene.get("image_metadata", [])
            if meta and isinstance(meta, list) and isinstance(meta[0], dict):
                img_w = meta[0].get("image_width") or 800
            
            mapper.map_single_scene_coordinates(scene, room_width, room_depth, img_w=img_w)

        # 2. Semantic Object Fusion & Deduplication
        all_objects = []
        for i, scene in enumerate(scenes):
            objs = scene.get("objects", [])
            for o in objs:
                # Track original view source index
                o["_view_source"] = i
                all_objects.append(o)

        fusion = ObjectFusion()
        merged_objects = fusion.fuse_objects(all_objects)
        merged_objects = _normalise_ids_list(merged_objects)

        # Base merge dictionary using the first scene
        base_scene = dict(scenes[0])

        # 3. Structural Merger for doors/windows
        struct_merger = StructuralMerger()
        doors, windows, furniture = struct_merger.merge_structures(scenes, merged_objects)
        base_scene["doors"] = doors
        base_scene["windows"] = windows
        base_scene["objects"] = furniture

        # 4. Room Coordinates Mapping (Final update & Door/Window projections)
        room_w, room_d, room_h = mapper.map_room_coordinates(base_scene, doors, windows)

        # 5. Occupancy Grid and Free Space Detection
        detector = FreeSpaceDetector(resolution_cm=5)
        grid = detector.build_occupancy_grid(room_w, room_d, base_scene["objects"])
        empty_regions, free_polygons = detector.detect_free_regions(grid, room_w, room_d)
        
        base_scene["occupancy_grid"] = grid
        base_scene["empty_regions"] = empty_regions
        base_scene["free_space_polygons"] = free_polygons

        # Recalculate merged free space analysis
        cells = grid.get("cells", []) if isinstance(grid, dict) else []
        free_cells = sum(row.count(0) for row in cells)
        total_cells = sum(len(row) for row in cells) if cells else 1
        free_pct = round((free_cells / total_cells) * 100, 1) if total_cells > 0 else 0.0
        res = grid.get("grid_resolution_cm", 5) if isinstance(grid, dict) else 5
        usable_area_sqm = round((free_cells * (res ** 2)) / 10000.0, 2)
        side_cm = int((usable_area_sqm ** 0.5) * 100)
        
        base_scene["free_space_analysis"] = {
            "percentage": free_pct,
            "usable_area_sqm": usable_area_sqm,
            "largest_rectangle": {
                "width_cm": side_cm,
                "depth_cm": side_cm
            },
            "free_space_confidence": 0.72
        }
        
        # Backward compatibility for polygon mapping
        base_scene["free_regions"] = [{"polygon": [[pt["x"], pt["y"]] for pt in poly]} for poly in free_polygons]

        # 5. Spatial Relationships Calculation
        img_w = 800
        meta = base_scene.get("image_metadata", [])
        if meta and isinstance(meta, list) and isinstance(meta[0], dict):
            img_w = meta[0].get("image_width") or 800
        relationships = mapper.generate_spatial_relationships(base_scene["objects"], doors, windows, room_w, room_d, img_w)
        base_scene["spatial_relationships"] = relationships

        # 6. Metadata average and merges
        coverage_vals = [s.get("scene_coverage_percentage") for s in scenes if s.get("scene_coverage_percentage") is not None]
        if coverage_vals:
            base_scene["scene_coverage_percentage"] = int(sum(coverage_vals) / len(coverage_vals))

        for key in ("layout_problems", "safety_analysis", "candidate_furniture_locations", "image_metadata", "image_quality"):
            combined = []
            seen_ids = set()
            for scene in scenes:
                items = scene.get(key, [])
                for item in items:
                    item_id = item.get("id") or str(item)
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        combined.append(item)
            base_scene[key] = combined

        # Serialize outputs to the session path if provided
        if session_path is not None:
            # Save scene graph
            nodes = [obj["id"] for obj in base_scene["objects"]]
            for d in base_scene["doors"]:
                nodes.append(d["id"])
            for w in base_scene["windows"]:
                nodes.append(w["id"])

            scene_graph_data = {
                "nodes": nodes,
                "edges": base_scene["spatial_relationships"]
            }
            save_json(scene_graph_data, session_path / "scene_graph.json")
            save_json(grid, session_path / "occupancy_grid.json")

        return base_scene

    # -------------------------------------------------------------------------
    # Helper: Memory Reclamation
    # -------------------------------------------------------------------------
    def _clear_memory(self) -> None:
        """Empties garbage collection and cleans CUDA caches."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug("Failed to release CUDA locks: %s", str(e))
