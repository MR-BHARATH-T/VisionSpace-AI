"""Image Preprocessing Module for VisionSpace AI.

Handles image loading, URL downloading with concurrent threads, image validation,
EXIF transposition, context safety bounds, and image count capping.
"""

from __future__ import annotations

import io
import logging
import pathlib
import time
import threading
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.registe
    r_heif_opener()
    logging.getLogger("VisionSpaceAI.ImagePreprocessing").info("pillow_heif registered successfully.")
except ImportError:
    logging.getLogger("VisionSpaceAI.ImagePreprocessing").debug("pillow_heif not available. HEIC support may be limited.")

import config

logger = logging.getLogger("VisionSpaceAI.ImagePreprocessing")

# Safe configuration settings resolution
MAX_IMAGE_SIZE = getattr(config, "MAX_IMAGE_SIZE", 1024)
MAX_IMAGES = getattr(config, "MAX_IMAGES", 8)


class VisionImagePreprocessor:
    """Class responsible for fetching, validating, and preprocessing images for the vision model."""

    def __init__(self) -> None:
        """Initializes the VisionImagePreprocessor."""
        self.last_resize_time = 0.0
        logger.info("VisionImagePreprocessor initialized with MAX_IMAGE_SIZE=%d, MAX_IMAGES=%d", MAX_IMAGE_SIZE, MAX_IMAGES)


    @staticmethod
    def _is_url(path: str) -> bool:
        """Checks if a string is an HTTP/HTTPS URL.

        Args:
            path: String path to check.

        Returns:
            True if it starts with http:// or https://.
        """
        return path.startswith("http://") or path.startswith("https://")

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        """Downloads an image from a URL with retries.

        Args:
            url: The HTTP/HTTPS URL of the image.

        Returns:
            A PIL Image.
        """
        logger.info("Downloading image from URL: %s", url)
        # Retry up to 3 times before failing
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30.0)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                logger.warning("Download attempt %d failed for %s: %s", attempt + 1, url, str(e))
                if attempt == 2:
                    raise e
                time.sleep(1.0)
        raise RuntimeError(f"Failed to download image from {url}")

    @staticmethod
    def _load_local_image(path: Union[str, pathlib.Path]) -> Image.Image:
        """Loads a local image file.

        Args:
            path: The local filesystem path.

        Returns:
            A PIL Image.
        """
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Local image not found: {p}")
        return Image.open(p)

    @staticmethod
    def _validate_image(image: Image.Image) -> Image.Image:
        """Validates that an image object is correct, has positive dimensions, and converts to RGB.

        Handles transparent modes (RGBA, LA, P) via alpha compositing and other modes appropriately.

        Args:
            image: The PIL Image to validate.

        Returns:
            A valid RGB PIL Image.
        """
        if image is None:
            raise ValueError("Image object is None.")
        if not isinstance(image, Image.Image):
            raise TypeError("Object is not a PIL Image instance.")
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Invalid image dimensions: {image.width}x{image.height}. Dimensions must be positive.")

        # Handle conversions and transparent images nicely
        if image.mode in ("RGBA", "LA", "P"):
            logger.debug("Converting transparent image mode %s to RGB via alpha composite", image.mode)
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        elif image.mode != "RGB":
            logger.debug("Converting image mode %s to RGB", image.mode)
            image = image.convert("RGB")

        return image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resizes an image if its largest side exceeds MAX_IMAGE_SIZE.

        Args:
            image: PIL Image.

        Returns:
            Resized or original PIL Image.
        """
        t0 = time.perf_counter()
        width, height = image.size
        largest_side = max(width, height)
        if largest_side > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / largest_side
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            logger.info(
                "Resizing oversized image from %dx%d to %dx%d (MAX_IMAGE_SIZE=%d)",
                width, height, new_width, new_height, MAX_IMAGE_SIZE
            )
            res = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.last_resize_time += time.perf_counter() - t0
            return res
        self.last_resize_time += time.perf_counter() - t0
        return image


    def _prepare_single_image(self, image: Union[str, pathlib.Path, Image.Image]) -> Image.Image:
        """Loads, validates, and resizes a single image input, correcting orientation with EXIF.

        Args:
            image: A PIL Image, path to a local file, or a URL string.

        Returns:
            Preprocessed RGB PIL Image.
        """
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, pathlib.Path):
            img = self._load_local_image(image)
        elif isinstance(image, str):
            img = self._download_image(image) if self._is_url(image) else self._load_local_image(image)
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Clean image orientation using EXIF transposition before validation/resizing
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as exif_err:
            logger.warning("EXIF transpose failed: %s", str(exif_err))

        return self._resize_image(self._validate_image(img))

    def _download_images_parallel(self, urls: List[str]) -> List[Image.Image]:
        """Downloads multiple image URLs concurrently using parallel threads.

        Args:
            urls: List of image URLs to download.

        Returns:
            List of downloaded PIL Images.
        """
        results: List[Optional[Image.Image]] = [None] * len(urls)
        errors: List[Optional[Exception]] = [None] * len(urls)

        def worker(idx: int, url: str) -> None:
            try:
                results[idx] = self._download_image(url)
            except Exception as e:
                errors[idx] = e

        threads = []
        for idx, url in enumerate(urls):
            t = threading.Thread(target=worker, args=(idx, url))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for err in errors:
            if err is not None:
                raise err

        return [r for r in results if r is not None]

    def _manage_context_size(self, images: List[Image.Image]) -> List[Image.Image]:
        """Estimates cumulative pixels and resizes images if they exceed safe VRAM budgets.

        Args:
            images: List of PIL Images.

        Returns:
            List of preprocessed and potentially resized PIL Images.
        """
        if not images:
            return images

        total_pixels = sum(img.width * img.height for img in images)
        pixel_budget = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE * 5

        if total_pixels > pixel_budget:
            t0 = time.perf_counter()
            logger.info(
                "Context protection activated: Cumulative pixels %d exceeds budget %d. Resizing inputs.",
                total_pixels, pixel_budget
            )
            ratio = (pixel_budget / total_pixels) ** 0.5
            resized_images = []
            for img in images:
                new_w = max(16, int(img.width * ratio))
                new_h = max(16, int(img.height * ratio))
                resized_images.append(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
            self.last_resize_time += time.perf_counter() - t0
            return resized_images

        return images

    def prepare_images(
        self,
        images: Union[Image.Image, str, pathlib.Path, List[Union[Image.Image, str, pathlib.Path]]],
    ) -> List[Image.Image]:
        """Converts diverse image inputs into preprocessed RGB images, using parallel async downloads.

        Enforces image cap limits.

        Args:
            images: A single image or a list of mixed image inputs.

        Returns:
            List of preprocessed PIL Images.
        """
        self.last_resize_time = 0.0
        if images is None:
            return []

        raw_list = images if isinstance(images, list) else [images]


        # Automatic Image Count Limit Capping
        if len(raw_list) > MAX_IMAGES:
            logger.warning(
                "Uploaded image count (%d) exceeds limit (%d). Sampling the first %d images.",
                len(raw_list), MAX_IMAGES, MAX_IMAGES
            )
            raw_list = raw_list[:MAX_IMAGES]

        urls = []
        url_positions = []
        others: List[Any] = [None] * len(raw_list)

        for idx, img in enumerate(raw_list):
            if isinstance(img, str) and self._is_url(img):
                urls.append(img)
                url_positions.append(idx)
            else:
                others[idx] = img

        # Asynchronously download URLs
        downloaded = []
        if urls:
            downloaded = self._download_images_parallel(urls)

        # Merge results
        for idx, img in zip(url_positions, downloaded):
            others[idx] = img

        processed_images = []
        for index, img in enumerate(others):
            try:
                processed_images.append(self._prepare_single_image(img))
            except Exception as e:
                logger.error("Failed to process image at index %d: %s", index, str(e))
                raise e

        # VRAM Context Protection resizing
        return self._manage_context_size(processed_images)

    def get_image_info(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Extracts metadata dimensions and mode information from a list of images.

        Args:
            images: List of PIL Images.

        Returns:
            List of dictionary metadata per image.
        """
        return [
            {"index": idx, "width": img.width, "height": img.height, "mode": img.mode}
            for idx, img in enumerate(images)
        ]
