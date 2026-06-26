"""Base Provider Module for VisionSpace AI.

Defines the contract interface that all backend model execution providers must implement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Generator

class BaseProvider:
    """Interface class that all model providers must implement."""

    def load(self, model_name: str, task: str, config_kwargs: Dict[str, Any]) -> None:
        """Loads model weights, tokenizers, and configuration parameters.

        Args:
            model_name: The name/identifier of the model on the backend.
            task: Task key (e.g. 'vision', 'reasoning').
            config_kwargs: Key-value dictionary of additional parameters.
        """
        raise NotImplementedError()

    def unload(self) -> None:
        """Unloads model components from device to free hardware resources."""
        raise NotImplementedError()

    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Generates text completion based on prompt instructions.

        Returns:
            Dictionary containing structured output fields or streaming generator.
        """
        raise NotImplementedError()

    def gpu_info(self) -> Dict[str, Any]:
        """Gathers properties of active GPU hardware."""
        raise NotImplementedError()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Gets diagnostics detailing GPU memory utilization."""
        raise NotImplementedError()

    def get_model_info(self) -> Dict[str, Any]:
        """Gathers config settings and capabilities of active model."""
        raise NotImplementedError()

    def get_processor_info(self) -> Dict[str, Any]:
        """Returns processor class name and tokenizer vocabulary properties."""
        raise NotImplementedError()

    def stop_generation(self) -> None:
        """Signals active generation thread to halt execution."""
        raise NotImplementedError()

    def count_tokens(self, text: Union[str, List[Dict[str, Any]]]) -> int:
        """Returns the tokens count of a text string or list of messages."""
        raise NotImplementedError()

    def estimate_cost(self, tokens: int, mode: str = "input") -> float:
        """Estimates model query prices based on token counts."""
        raise NotImplementedError()

    def remaining_context(self, current_tokens: int) -> int:
        """Returns remaining headroom tokens count."""
        raise NotImplementedError()

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """Lists downloaded model repositories stored in the Cache."""
        raise NotImplementedError()

    def clear_cache(self) -> bool:
        """Clears local Cache directories."""
        raise NotImplementedError()

    def cache_size(self) -> float:
        """Gets total disk storage occupied by the Cache in GB."""
        raise NotImplementedError()
