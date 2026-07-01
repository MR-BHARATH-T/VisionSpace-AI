# utils/llm_factory.py
"""
VisionSpace AI — Production LLM Factory
=========================================
Model-agnostic factory for creating model instances and provider
configurations. Acts as a thin shim between application code and
ModelManager, keeping the rest of the codebase decoupled from
implementation details.

Responsibilities:
  - Validate and resolve task → model config from MODEL_REGISTRY
  - Build provider-specific loading kwargs (quantization, device, dtype)
  - Expose helper utilities: ModelFactory, ModelConfig, build_messages
  - Forward all inference to ModelManager (single point of truth)
  - Support multi-backend future extensibility (HuggingFace, OpenAI, vLLM)

Usage::

    from utils.llm_factory import ModelFactory

    factory = ModelFactory()
    response = factory.generate(
        task="vision",
        messages=[{"role": "user", "content": "Describe this room."}],
        images=[pil_image],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import config
from utils.constants import TASK_KEYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "vision": {
        "model":     getattr(config, "MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        "loader":    AutoModelForImageTextToText,
        "processor": AutoProcessor,
        "task_type": "vision",
        "modalities": ["text", "image"],
        "max_context": 32768,
    },
    "reasoning": {
        "model":     getattr(config, "MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        "loader":    AutoModelForImageTextToText,
        "processor": AutoProcessor,
        "task_type": "text",
        "modalities": ["text"],
        "max_context": 32768,
    },
}

# Task aliases: maps agent task names to registry keys
TASK_ALIAS: dict[str, str] = {
    "vision":        "vision",
    "spatial":       "reasoning",
    "design":        "reasoning",
    "commerce":      "reasoning",
    "budget":        "reasoning",
    "explain":       "reasoning",
    "visualization": "vision",   # future: dedicated inpainting model
    "text":          "reasoning",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class ModelProfile(str, Enum):
    """Speed/accuracy trade-off profile."""
    FAST     = "Fast"
    BALANCED = "Balanced"
    ACCURATE = "Accurate"


class Backend(str, Enum):
    """Inference backend selector."""
    HUGGINGFACE = "huggingface"
    OPENAI      = "openai"
    VLLM        = "vllm"


@dataclass
class ModelConfig:
    """Resolved configuration for a single model task.

    Attributes:
        task:          Logical task name (e.g. "vision", "reasoning").
        model_id:      Hugging Face model ID string.
        loader:        Model class (e.g. AutoModelForImageTextToText).
        processor:     Processor class (e.g. AutoProcessor).
        device:        Torch device string.
        dtype:         Torch dtype.
        quantization:  Quantization strategy ("AUTO", "4bit", "8bit", "none").
        profile:       Speed/accuracy profile.
        max_context:   Maximum token context window.
        modalities:    Supported input modalities.
        extra_kwargs:  Additional loader kwargs.
    """
    task:          str
    model_id:      str
    loader:        Any
    processor:     Any
    device:        str                    = field(default_factory=lambda: config.DEVICE)
    dtype:         torch.dtype            = field(default_factory=lambda: config.DTYPE)
    quantization:  str                    = field(default_factory=lambda: config.QUANTIZATION)
    profile:       ModelProfile           = ModelProfile.BALANCED
    max_context:   int                    = 32768
    modalities:    list[str]              = field(default_factory=lambda: ["text"])
    extra_kwargs:  dict[str, Any]         = field(default_factory=dict)

    def to_loader_kwargs(self) -> dict:
        """Build kwargs dict suitable for passing to the model loader."""
        kwargs: dict[str, Any] = {
            "cache_dir":       config.HF_CACHE_DIR,
            "revision":        config.HF_REVISION,
            "trust_remote_code": True,
        }

        # Dtype
        if self.device == "cuda":
            kwargs["torch_dtype"] = self.dtype
        else:
            kwargs["torch_dtype"] = torch.float32

        # Quantization
        if self.quantization in ("4bit", "AUTO") and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning("BitsAndBytes not available — skipping 4-bit quantization.")
        elif self.quantization == "8bit" and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("BitsAndBytes not available — skipping 8-bit quantization.")

        # CPU offloading for low-VRAM scenarios
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "cpu"

        kwargs.update(self.extra_kwargs)
        return kwargs

    def to_processor_kwargs(self) -> dict:
        """Build kwargs dict suitable for the processor loader."""
        return {
            "cache_dir":         config.HF_CACHE_DIR,
            "revision":          config.HF_REVISION,
            "trust_remote_code": True,
        }


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Model-agnostic factory for VisionSpace AI model access.

    Thin wrapper around ModelManager that resolves task → ModelConfig and
    provides a clean interface for all agent code.

    All actual model loading and inference is delegated to ModelManager.

    Args:
        backend: Inference backend (only ``huggingface`` is supported now).
    """

    def __init__(self, backend: Backend = Backend.HUGGINGFACE) -> None:
        self._backend = backend
        self._manager = None   # Lazy import

    # ------------------------------------------------------------------
    # Config Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_config(
        task: str,
        profile: ModelProfile = ModelProfile.BALANCED,
    ) -> ModelConfig:
        """Resolve a task name to a :class:`ModelConfig`.

        Args:
            task:    Logical task name (vision, reasoning, spatial, …).
            profile: Speed/accuracy profile.

        Returns:
            Populated :class:`ModelConfig` instance.

        Raises:
            ValueError: If *task* cannot be resolved.
        """
        registry_key = TASK_ALIAS.get(task.lower())
        if registry_key is None:
            available = list(TASK_ALIAS.keys())
            raise ValueError(
                f"Unknown task '{task}'. Available: {available}"
            )

        registry_entry = MODEL_REGISTRY.get(registry_key)
        if registry_entry is None:
            raise ValueError(
                f"Registry key '{registry_key}' not found in MODEL_REGISTRY."
            )

        return ModelConfig(
            task=registry_key,
            model_id=registry_entry["model"],
            loader=registry_entry["loader"],
            processor=registry_entry["processor"],
            profile=profile,
            max_context=registry_entry.get("max_context", 32768),
            modalities=registry_entry.get("modalities", ["text"]),
        )

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def generate(
        self,
        task: str,
        messages: list[dict],
        images: list[Any] | None = None,
        stream_callback: Callable | None = None,
        request_id: str | None = None,
        **generation_kwargs,
    ) -> dict:
        """Generate a response for the given task and messages.

        Delegates to :class:`~models.model_manager.ModelManager`.

        Args:
            task:             Logical task name (vision, reasoning, …).
            messages:         Chat-format message list.
            images:           Optional PIL images (for vision tasks).
            stream_callback:  Optional ``on_token(str)`` callback for streaming.
            request_id:       Optional request ID for cancellation.
            **generation_kwargs: Passed to the model (temperature, max_new_tokens …).

        Returns:
            Dict with ``text``, ``request_id``, ``task``, ``duration_s``.
        """
        manager = self._get_manager()

        # Attach images to last user message if provided
        if images:
            messages = _inject_images(messages, images)

        return manager.generate(
            task=task,
            messages=messages,
            stream_callback=stream_callback,
            request_id=request_id,
            **generation_kwargs,
        )

    async def generate_async(
        self,
        task: str,
        messages: list[dict],
        images: list[Any] | None = None,
        **generation_kwargs,
    ):
        """Async wrapper — delegates to ModelManager.generate_async()."""
        manager = self._get_manager()
        if images:
            messages = _inject_images(messages, images)
        return await manager.generate_async(task=task, messages=messages, **generation_kwargs)

    # ------------------------------------------------------------------
    # Cache / Model Management
    # ------------------------------------------------------------------

    def preload(self, task: str) -> None:
        """Preload (warm up) a model for the given task.

        Args:
            task: Task name to preload.
        """
        manager = self._get_manager()
        registry_key = TASK_ALIAS.get(task.lower(), task)
        logger.info("ModelFactory: preloading model for task '%s' (key='%s')", task, registry_key)
        manager.preload(registry_key)

    def unload(self, task: str) -> None:
        """Unload the model for the given task to free memory."""
        manager = self._get_manager()
        registry_key = TASK_ALIAS.get(task.lower(), task)
        manager.unload(registry_key)

    def model_status(self) -> dict:
        """Return status report from ModelManager."""
        return self._get_manager().status_report()

    def clear_cache(self) -> None:
        """Clear the Hugging Face model cache via ModelManager."""
        manager = self._get_manager()
        if hasattr(manager, "cache_manager"):
            manager.cache_manager.clear_cache()
        else:
            logger.warning("No cache_manager attached to ModelManager.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_manager(self):
        if self._manager is None:
            from models.model_manager import ModelManager
            self._manager = ModelManager.get_instance()
        return self._manager


# ---------------------------------------------------------------------------
# Message Builder Utilities
# ---------------------------------------------------------------------------

def build_messages(
    system: str,
    user: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Construct a chat-format message list.

    Args:
        system:  System prompt string.
        user:    User turn content.
        history: Optional prior turns to prepend (list of role/content dicts).

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def build_vision_messages(
    system: str,
    user: str,
    images: list[Any],
) -> list[dict]:
    """Build a message list with inline image references for vision models.

    Images are embedded as ``{"type": "image_url", "image_url": …}`` parts
    in the user content — compatible with Qwen-VL / GPT-4V style APIs.

    Args:
        system: System prompt.
        user:   Text instruction for the vision turn.
        images: List of PIL Images or URL strings.

    Returns:
        Chat message list with image content parts.
    """
    import base64
    import io

    image_parts = []
    for img in images:
        if isinstance(img, str):
            # Assume URL
            image_parts.append({
                "type":      "image_url",
                "image_url": {"url": img},
            })
        else:
            # PIL Image → base64
            try:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                image_parts.append({
                    "type":      "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except Exception as exc:
                logger.warning("Could not encode image: %s", exc)

    user_content = image_parts + [{"type": "text", "text": user}]
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def _inject_images(messages: list[dict], images: list[Any]) -> list[dict]:
    """Attach *images* to the last user message in *messages*.

    Converts the last user message's ``content`` from a plain string to a
    list of content parts including image placeholders recognised by
    HuggingFace vision processors.

    Args:
        messages: Existing message list.
        images:   PIL images or URL strings to attach.

    Returns:
        Updated message list (copy, original unmodified).
    """
    messages = [dict(m) for m in messages]   # shallow copy
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            existing = messages[i]["content"]
            if isinstance(existing, str):
                messages[i]["content"] = [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text",  "text": existing},
                ]
            elif isinstance(existing, list):
                messages[i]["content"] = [
                    *[{"type": "image", "image": img} for img in images],
                    *existing,
                ]
            break
    return messages


# ---------------------------------------------------------------------------
# Module-Level Singleton
# ---------------------------------------------------------------------------

_default_factory: ModelFactory | None = None


def get_factory() -> ModelFactory:
    """Return the module-level singleton :class:`ModelFactory`."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ModelFactory()
    return _default_factory


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "MODEL_REGISTRY",
    "TASK_ALIAS",
    "ModelProfile",
    "Backend",
    "ModelConfig",
    "ModelFactory",
    "build_messages",
    "build_vision_messages",
    "get_factory",
]
