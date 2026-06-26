"""Vision Model Facade for VisionSpace AI.

Client facade class that coordinates image prep, model loading, and routes
generations through the central ModelManager orchestrator.
"""

from __future__ import annotations

import logging
import pathlib
import concurrent.futures
from typing import Any, Dict, List, Optional, Union, Generator

from PIL import Image

from utils.image_preprocessing import VisionImagePreprocessor
from models.model_manager import ModelManager

logger = logging.getLogger("VisionSpaceAI.VisionModel")


class VisionModel:
    """Facade vision model class coordinating image preprocessing and inference via ModelManager."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the VisionModel facade structure and references ModelManager."""
        self.provider_name = provider
        self.manager = ModelManager()
        self.provider = self.manager._get_provider(provider, "vision")

        # Preprocessor subsystem
        self.preprocessor = VisionImagePreprocessor()

        # Hooks forwarding
        self.before_generation = None
        self.after_generation = None
        self.before_loading = None
        self.after_loading = None

        self.on_model_loading = None
        self.on_model_loaded = None
        self.on_download_started = None
        self.on_download_finished = None

    def _sync_callbacks(self) -> None:
        """Propagates callbacks to the provider instance."""
        self.provider.before_generation = self.before_generation
        self.provider.after_generation = self.after_generation
        self.provider.before_loading = self.before_loading
        self.provider.after_loading = self.after_loading

        self.provider.on_model_loading = self.on_model_loading
        self.provider.on_model_loaded = self.on_model_loaded
        self.provider.on_download_started = self.on_download_started
        self.provider.on_download_finished = self.on_download_finished

    @property
    def is_loaded(self) -> bool:
        """Checks if the model parameters are loaded."""
        return self.manager.is_loaded("vision")

    def load(self) -> None:
        """Loads processor and model weights via the orchestrator."""
        self._sync_callbacks()
        self.manager.load_model("vision")

    def unload(self) -> None:
        """Unloads the model and processor resources."""
        self.manager.unload_model("vision")

    def reload(self) -> None:
        """Reloads the model weights."""
        self.unload()
        self.load()

    def cleanup(self) -> None:
        """Alias for unload."""
        self.unload()

    def health_check(self) -> bool:
        """Performs a wellness check running a 1x1 dummy image generation request."""
        return self.manager.health_check("vision")

    def stop_generation(self) -> None:
        """Signals active generation thread to halt execution."""
        self.provider.stop_generation()

    @property
    def streaming_stats(self) -> Dict[str, Any]:
        """Gets real-time token/sec and elapsed metrics of the streaming generation."""
        return self.provider.streaming_stats

    @staticmethod
    def build_messages(
        system: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Constructs unified chat messages aggregates."""
        from providers.huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider.build_messages(
            system=system, user=user, assistant=assistant, history=history
        )

    # =========================================================================
    # Token utilities
    # =========================================================================
    def count_tokens(self, text: Union[str, List[Dict[str, Any]]]) -> int:
        """Returns the token counts for text or message list."""
        return self.manager.count_tokens("vision", text)

    def max_context(self) -> int:
        """Returns maximum allowed context length."""
        return self.manager.max_context("vision")

    def estimate_cost(self, tokens: int, mode: str = "input") -> float:
        """Estimates model query prices based on token counts."""
        return self.provider.estimate_cost(tokens, mode)

    def remaining_context(self, current_tokens: int) -> int:
        """Returns remaining headroom tokens count."""
        return self.manager.remaining_context("vision", current_tokens)

    # =========================================================================
    # Cache management forwarding
    # =========================================================================
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """Lists downloaded model repositories stored in the Cache."""
        return self.manager.list_models()

    def clear_cache(self) -> bool:
        """Clears local Cache directories."""
        return self.manager.clear_cache()

    def cache_size(self) -> float:
        """Gets total disk storage occupied by the Cache in GB."""
        return self.manager.cache_size()

    # =========================================================================
    # GPU Diagnostics Facades
    # =========================================================================
    def gpu_info(self) -> Dict[str, Any]:
        """Gathers properties of active GPU hardware."""
        return self.provider.gpu_info()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Gets diagnostics detailing GPU memory utilization."""
        return self.manager.get_memory_usage()

    def get_model_info(self) -> Dict[str, Any]:
        """Gathers settings and capabilities of active model."""
        info = self.manager.get_model_info("vision")
        info["family"] = info.get("model_name", "Qwen")
        info["version"] = "2.5"
        return info

    def get_processor_info(self) -> Dict[str, Any]:
        """Returns processor class name and tokenizer vocabulary properties."""
        return self.provider.get_processor_info()

    # =========================================================================
    # Preprocessor Delegations (Backwards-compatibility support)
    # =========================================================================
    def prepare_images(
        self,
        images: Union[Image.Image, str, pathlib.Path, List[Union[Image.Image, str, pathlib.Path]]],
    ) -> List[Image.Image]:
        """Converts diverse image inputs into preprocessed RGB images."""
        return self.preprocessor.prepare_images(images)

    def get_image_info(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Extracts metadata dimensions and mode information from a list of images."""
        return self.preprocessor.get_image_info(images)

    # =========================================================================
    # Prompts & Input Preparation
    # =========================================================================
    def prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Renders messages and image tokens to PyTorch tensors."""
        if not self.is_loaded:
            self.load()
        return self.provider.prepare_inputs(messages, images=images, videos=videos)

    # =========================================================================
    # Inference Routing Facades
    # =========================================================================
    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[Union[Image.Image, str, pathlib.Path, List[Union[Image.Image, str, pathlib.Path]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stream: bool = False,
        profile: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Union[Dict[str, Any], Any]:
        """Runs model inference routing query to ModelManager orchestrator."""
        self._sync_callbacks()

        # Preprocess images if present
        processed_images = self.prepare_images(images) if images is not None else []

        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if do_sample is not None:
            kwargs["do_sample"] = do_sample
        if profile is not None:
            kwargs["profile"] = profile
        if response_format is not None:
            kwargs["response_format"] = response_format

        kwargs["stream"] = stream
        kwargs["images"] = processed_images

        if stream:
            return self.manager.generate_stream(
                task="vision",
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )
        else:
            return self.manager.generate(
                task="vision",
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )

    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[Optional[Union[Image.Image, str, pathlib.Path, List[Union[Image.Image, str, pathlib.Path]]]]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """Runs batch generation via ModelManager."""
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if do_sample is not None:
            kwargs["do_sample"] = do_sample

        # Preprocess list of images if list matches prompts
        processed_images_list = []
        if images_list is not None:
            for imgs in images_list:
                processed_images_list.append(self.prepare_images(imgs) if imgs is not None else None)
            kwargs["images_list"] = processed_images_list

        return self.manager.generate_batch("vision", prompts, **kwargs)

    def generate_async(
        self,
        prompt: Optional[str] = None,
        images: Optional[Union[Image.Image, str, pathlib.Path, List[Union[Image.Image, str, pathlib.Path]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> concurrent.futures.Future:
        """Asynchronously runs generation via central orchestrator executor."""
        self._sync_callbacks()
        # Preprocess image
        if images is not None:
            kwargs["images"] = self.prepare_images(images)
        kwargs["prompt"] = prompt
        kwargs["messages"] = messages
        kwargs["system_prompt"] = system_prompt
        return self.manager.generate_async("vision", **kwargs)