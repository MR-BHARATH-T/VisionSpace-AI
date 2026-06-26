"""Text Model Facade for VisionSpace AI.

Client facade class that routes text reasoning generations and diagnostic metrics
directly to the central ModelManager orchestrator.
"""

from __future__ import annotations

import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, Union, Generator

from models.model_manager import ModelManager

logger = logging.getLogger("VisionSpaceAI.TextModel")


class TextModel:
    """Stable public facade class wrapping Text Model interactions via the ModelManager."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the TextModel referencing the central ModelManager orchestrator."""
        self.provider_name = provider
        self.manager = ModelManager()
        self.provider = self.manager._get_provider(provider, "reasoning")

        # Hook placeholders
        self.before_generation = None
        self.after_generation = None
        self.before_loading = None
        self.after_loading = None

        self.on_model_loading = None
        self.on_model_loaded = None
        self.on_download_started = None
        self.on_download_finished = None

    def _sync_callbacks(self) -> None:
        """Propagates registered callbacks to the active provider instance."""
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
        return self.manager.is_loaded("reasoning")

    def load(self) -> None:
        """Triggers model weights loading via the orchestrator."""
        self._sync_callbacks()
        self.manager.load_model("reasoning")

    def unload(self) -> None:
        """Unloads model components from device."""
        self.manager.unload_model("reasoning")

    def reload(self) -> None:
        """Reloads the model weights."""
        self.unload()
        self.load()

    def cleanup(self) -> None:
        """Cleans up active references."""
        self.unload()

    def health_check(self) -> bool:
        """Runs validation test to check model integrity."""
        return self.manager.health_check("reasoning")

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
        """Returns the tokens count of a text string or list of messages."""
        return self.manager.count_tokens("reasoning", text)

    def max_context(self) -> int:
        """Returns maximum positions context bounds allowed by the model."""
        return self.manager.max_context("reasoning")

    def estimate_cost(self, tokens: int, mode: str = "input") -> float:
        """Estimates model query prices based on token counts."""
        return self.provider.estimate_cost(tokens, mode)

    def remaining_context(self, current_tokens: int) -> int:
        """Returns remaining headroom tokens count."""
        return self.manager.remaining_context("reasoning", current_tokens)

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
        """Gathers config settings and capabilities of active model."""
        info = self.manager.get_model_info("reasoning")
        info["family"] = info.get("model_name", "Qwen")
        info["version"] = "3.0"
        return info

    def get_processor_info(self) -> Dict[str, Any]:
        """Returns processor class name and tokenizer vocabulary properties."""
        return self.provider.get_processor_info()

    # =========================================================================
    # Generation & Streaming Facades
    # =========================================================================
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stream: bool = False,
        profile: Optional[str] = None,
        response_format: Optional[str] = None,
        prompt_file: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        validator: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Inference routing facade. Generates output text predictions via ModelManager."""
        self._sync_callbacks()

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
        if prompt_file is not None:
            kwargs["prompt_file"] = prompt_file
        if variables is not None:
            kwargs["variables"] = variables
        if validator is not None:
            kwargs["validator"] = validator

        kwargs["stream"] = stream

        if stream:
            return self.manager.generate_stream(
                task="reasoning",
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )
        else:
            return self.manager.generate(
                task="reasoning",
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """Runs multiple prompt inferences via queue worker thread."""
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if do_sample is not None:
            kwargs["do_sample"] = do_sample

        return self.manager.generate_batch("reasoning", prompts, **kwargs)

    def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> concurrent.futures.Future:
        """Asynchronously runs generation via central orchestrator executor."""
        self._sync_callbacks()
        kwargs["prompt"] = prompt
        kwargs["messages"] = messages
        kwargs["system_prompt"] = system_prompt
        return self.manager.generate_async("reasoning", **kwargs)
