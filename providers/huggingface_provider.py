"""Hugging Face Provider Module for VisionSpace AI.

Coordinates cache scanning, warm pool memory limits, prompt optimization, background queues,
and generation stopping criteria.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import queue
import hashlib
import threading
import copy
import pathlib
from typing import Any, Dict, List, Optional, Union, Generator

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)
from transformers.utils import is_flash_attn_2_available
from transformers import StoppingCriteria, StoppingCriteriaList
from huggingface_hub import scan_cache_dir

import config
from providers.base_provider import BaseProvider
from utils.json_parser import extract_and_validate_json, validate_response_rules

logger = logging.getLogger("VisionSpaceAI.HuggingFaceProvider")

# Safe config variables resolution
HF_VISION_MODEL = getattr(config, "HF_VISION_MODEL", None)
if HF_VISION_MODEL is None:
    models_dict = getattr(config, "MODELS", {})
    HF_VISION_MODEL = models_dict.get("vision", {}).get("model", "Qwen/Qwen2.5-VL-7B-Instruct")

HF_TEXT_MODEL = getattr(config, "HF_TEXT_MODEL", None)
if HF_TEXT_MODEL is None:
    models_dict = getattr(config, "MODELS", {})
    HF_TEXT_MODEL = models_dict.get("text", {}).get("model", "Qwen/Qwen3-VL-4B-Instruct")

MAX_NEW_TOKENS = getattr(config, "MAX_NEW_TOKENS", 2048)
TEMPERATURE = getattr(config, "TEMPERATURE", 0.2)
TOP_P = getattr(config, "TOP_P", 0.95)
DO_SAMPLE = getattr(config, "DO_SAMPLE", False)

# Device mapping
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Registry Task Mapping
MODEL_REGISTRY = {
    "vision": {
        "model": HF_VISION_MODEL,
        "loader": AutoModelForImageTextToText,
        "processor": AutoProcessor,
    },
    "reasoning": {
        "model": HF_TEXT_MODEL,
        "loader": AutoModelForImageTextToText,
        "processor": AutoProcessor,
    }
}


# Custom Stopping Criteria for Generation Cancellation
class CancelStoppingCriteria(StoppingCriteria):
    """Custom Hugging Face stopping criteria to interrupt generation instantly."""

    def __init__(self, check_fn) -> None:
        self.check_fn = check_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        return self.check_fn()


# Model Warm Pool Manager
class WarmPool:
    """Manages loaded models to keep them warm when VRAM permits."""

    def __init__(self) -> None:
        self.pool: Dict[str, Any] = {}
        self.load_order: List[str] = []
        self._lock = threading.Lock()

    def add(self, key: str, model: Any, tokenizer: Any, processor: Any) -> None:
        with self._lock:
            if key in self.pool:
                self.load_order.remove(key)
            self.pool[key] = {"model": model, "tokenizer": tokenizer, "processor": processor}
            self.load_order.append(key)
            self._prune()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self.pool:
                # Move to end of LRU list
                self.load_order.remove(key)
                self.load_order.append(key)
                return self.pool[key]
            return None

    def remove(self, key: str) -> None:
        with self._lock:
            if key in self.pool:
                logger.info("Warm Pool: removing model %s to reclaim resources", key)
                entry = self.pool.pop(key)
                self.load_order.remove(key)
                del entry["model"]
                del entry["tokenizer"]
                del entry["processor"]
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                gc.collect()

    def _prune(self) -> None:
        """Automatically unloads oldest model if VRAM is low or we have more than 2 models."""
        if not torch.cuda.is_available():
            return
        try:
            # Prune by count first (keep max 2 models warm)
            while len(self.pool) > 2:
                oldest = self.load_order[0]
                logger.info("Warm Pool pruning oldest loaded model: %s", oldest)
                self.pool.pop(oldest)
                self.load_order.remove(oldest)

            # Check memory usage
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            free = total_vram - allocated
            if free < 4.0 and len(self.pool) > 1:
                oldest = self.load_order[0]
                logger.info("VRAM is low (%.2f GB free). Evicting %s from Warm Pool", free, oldest)
                self.pool.pop(oldest)
                self.load_order.remove(oldest)
        except Exception as e:
            logger.warning("Error pruning warm pool: %s", str(e))


GLOBAL_WARM_POOL = WarmPool()


def get_model_size_gb(model: Any) -> float:
    """Calculates parameters footprint size of the loaded model in Gigabytes."""
    if model is None:
        return 0.0
    try:
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return 0.0


def format_prompt(template: str, variables: Optional[Dict[str, Any]]) -> str:
    """Formats template text with variables using Jinja2 if available, falling back to basic replace."""
    if not variables:
        return template
    try:
        import jinja2
        template_obj = jinja2.Template(template)
        return template_obj.render(**variables)
    except Exception:
        result = template
        for k, v in variables.items():
            result = result.replace(f"{{{k}}}", str(v))
            result = result.replace(f"{{{{{k}}}}}", str(v))
        return result


# Helper function to compute raw GPU stats
def get_gpu_utilization() -> Dict[str, Any]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        util = rates.gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used
        mem_total = mem_info.total
        mem_percent = (mem_used / mem_total) * 100
        pynvml.nvmlShutdown()
        return {"gpu_utilization_percent": util, "gpu_memory_percent": round(mem_percent, 2)}
    except Exception:
        pass

    try:
        import subprocess
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd, shell=True, text=True).strip()
        parts = output.split(",")
        if len(parts) == 3:
            gpu_util, mem_used, mem_total = map(float, parts)
            mem_percent = (mem_used / mem_total) * 100
            return {"gpu_utilization_percent": gpu_util, "gpu_memory_percent": round(mem_percent, 2)}
    except Exception:
        pass

    return {"gpu_utilization_percent": "N/A", "gpu_memory_percent": "N/A"}


class HuggingFaceProvider(BaseProvider):
    """Production implementation of HuggingFace execution provider backend."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.model_name: str = ""
        self.task: str = ""
        self.loaded: bool = False
        self.max_context_length: int = 2048
        self.quantization_profile: str = "none"

        # Hooks and callbacks
        self.before_generation = None
        self.after_generation = None
        self.before_loading = None
        self.after_loading = None

        self.on_model_loading = None
        self.on_model_loaded = None
        self.on_download_started = None
        self.on_download_finished = None

        # Thread Safety & Cancellation
        self._lock = threading.Lock()
        self._stop_generation = False
        self._cancellation_token_fn = None

        # In-Memory Cache
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

        # Streaming stats
        self.streaming_stats = {
            "tokens_per_sec": 0.0,
            "elapsed": 0.0,
            "generated_tokens": 0,
            "remaining_estimate": 0.0
        }

    def set_cancellation_token(self, fn: Any) -> None:
        """Configures a callable cancellation token check function."""
        self._cancellation_token_fn = fn

    # =========================================================================
    # Hugging Face Cache Manager
    # =========================================================================
    @staticmethod
    def list_cached_models() -> List[Dict[str, Any]]:
        """Scans the local HF cache directory and returns detailed repository metadata."""
        cache_dir = getattr(config, "HF_CACHE_DIR", None)
        if not cache_dir or not os.path.exists(cache_dir):
            return []
        try:
            cache_info = scan_cache_dir(cache_dir=cache_dir)
            models = []
            for repo in cache_info.repos:
                models.append({
                    "repo_id": repo.repo_id,
                    "repo_type": repo.repo_type,
                    "size_on_disk_gb": round(repo.size_on_disk / (1024 ** 3), 2),
                    "size_on_disk_str": repo.size_on_disk_str,
                    "last_modified": repo.last_modified.isoformat() if repo.last_modified else "N/A"
                })
            return models
        except Exception as e:
            logger.error("Error scanning cache directory: %s", str(e))
            return []

    @staticmethod
    def clear_cache() -> bool:
        """Deletes the cache folder files safely to clean disk usage."""
        cache_dir = getattr(config, "HF_CACHE_DIR", None)
        if not cache_dir or not os.path.exists(cache_dir):
            return False
        try:
            import shutil
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.info("HF cache directory cleared successfully.")
            return True
        except Exception as e:
            logger.error("Failed to clear HF cache: %s", str(e))
            return False

    @staticmethod
    def cache_size() -> float:
        """Returns the total disk storage occupied by HF Cache in Gigabytes."""
        cache_dir = getattr(config, "HF_CACHE_DIR", None)
        if not cache_dir or not os.path.exists(cache_dir):
            return 0.0
        total = 0
        for dirpath, _, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return round(total / (1024 ** 3), 2)

    # =========================================================================
    # Loader contracts & Validation
    # =========================================================================
    def load(self, model_name: str, task: str, config_kwargs: Dict[str, Any] = None) -> None:
        if self.loaded and self.model_name == model_name:
            return

        with self._lock:
            if self.loaded and self.model_name == model_name:
                return

            if self.before_loading:
                self.before_loading()

            if self.on_model_loading:
                self.on_model_loading(model_name)

            self.model_name = model_name
            self.task = task

            # Check if model is cached in Warm Pool
            warm_entry = GLOBAL_WARM_POOL.get(model_name)
            if warm_entry:
                logger.info("Warm Pool: cache hit. Reusing loaded model '%s'", model_name)
                self.model = warm_entry["model"]
                self.tokenizer = warm_entry["tokenizer"]
                self.processor = warm_entry["processor"]
                self.loaded = True
                if self.on_model_loaded:
                    self.on_model_loaded(model_name)
                if self.after_loading:
                    self.after_loading()
                return

            print(f"[Loading] Loading/Downloading Hugging Face model config & weights for: {model_name}...")
            logger.info("HuggingFaceProvider loading '%s' for task '%s'", model_name, task)
            cache_dir = getattr(config, "HF_CACHE_DIR", None)
            auth_token = getattr(config, "HF_TOKEN", None)
            hf_revision = getattr(config, "HF_REVISION", None)
            local_files_only = getattr(config, "LOCAL_FILES_ONLY", False)
            force_download = getattr(config, "FORCE_DOWNLOAD", False)

            if self.on_download_started:
                self.on_download_started(model_name)

            # Config inspect & dynamic validation
            model_config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                token=auth_token,
                local_files_only=local_files_only,
                force_download=force_download,
                revision=hf_revision,
            )

            # Auto Model Validation (ensure supports text generation)
            architectures = getattr(model_config, "architectures", [])
            arch_name = architectures[0] if architectures else ""
            model_type = getattr(model_config, "model_type", "").lower()
            text_gen_indicators = ["causallm", "lmhead", "conditionalgeneration", "seq2seq", 
                                    "texttoxt", "vl", "imagetextto", "mplug", "llava", "internvl", "phi"]
            is_valid = any(ind in arch_name.lower() or ind in model_type for ind in text_gen_indicators)
            if not is_valid:
                raise ValueError(
                    f"Selected Hugging Face model '{model_name}' does not support text generation "
                    f"(architectures: {architectures}, type: {model_type})."
                )

            # Retrieve loader class from registry
            loader_entry = MODEL_REGISTRY.get(task, MODEL_REGISTRY["reasoning"])
            loader_class = loader_entry["loader"]
            processor_class = loader_entry["processor"]

            # Load tokenizer and processor (only if not already set by ModelManager)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    token=auth_token,
                    local_files_only=local_files_only,
                    force_download=force_download,
                    revision=hf_revision,
                )

            if self.processor is None:
                if processor_class is not None:
                    if not hasattr(self, "_processor_cache"):
                        self._processor_cache = {}
                    if model_name not in self._processor_cache:
                        print(f"[Processor] Instantiating processor configurations for: {model_name}...")
                        self._processor_cache[model_name] = processor_class.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            token=auth_token,
                            local_files_only=local_files_only,
                            force_download=force_download,
                            revision=hf_revision,
                        )
                    self.processor = self._processor_cache[model_name]
                else:
                    self.processor = None

            if self.on_download_finished:
                self.on_download_finished(model_name)
            print(f"[Success] Model weights loaded successfully: {model_name}")

            # Quantization profile resolution
            quant_profile = getattr(config, "QUANTIZATION", "none")
            if str(quant_profile).lower() == "auto":
                if DEVICE == "cuda":
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if total_vram_gb >= 78.0:
                        quant_profile = "fp16"
                    elif total_vram_gb >= 38.0:
                        quant_profile = "bf16"
                    elif total_vram_gb >= 15.0:
                        quant_profile = "4bit"
                    else:
                        quant_profile = "4bit"
                else:
                    quant_profile = "fp16"

            self.quantization_profile = quant_profile

            quant_config = None
            device_map = None
            model_dtype = getattr(config, "DTYPE", torch.float16)

            bf16_supported = False
            if DEVICE == "cuda":
                try:
                    bf16_supported = torch.cuda.is_bf16_supported()
                except Exception:
                    pass

            if DEVICE == "cuda":
                if quant_profile == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    device_map = "auto"
                elif quant_profile == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    device_map = "auto"
                elif quant_profile == "bf16":
                    model_dtype = torch.bfloat16 if bf16_supported else torch.float16
                    device_map = "auto"
                elif quant_profile == "fp16":
                    model_dtype = torch.float16
                    device_map = "auto"
            else:
                device_map = DEVICE
                if DEVICE == "cpu" or DEVICE == "mps":
                    model_dtype = torch.float32

            # Flash Attention dynamic detection
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

            # Execute load
            logger.info("Executing load weights for: %s (attn: %s, dtype: %s)", model_name, attn_impl, str(model_dtype))
            try:
                self.model = loader_class.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map=device_map,
                    torch_dtype=model_dtype,
                    attn_implementation=attn_impl,
                    cache_dir=cache_dir,
                    token=auth_token,
                    local_files_only=local_files_only,
                    force_download=force_download,
                    revision=hf_revision,
                )
            except Exception as e:
                logger.warning("Loading failed with %s implementation: %s. Retrying with SDPA.", attn_impl, str(e))
                self.model = loader_class.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    device_map=device_map,
                    torch_dtype=model_dtype,
                    attn_implementation="sdpa",
                    cache_dir=cache_dir,
                    token=auth_token,
                    local_files_only=local_files_only,
                    force_download=force_download,
                    revision=hf_revision,
                )

            # Context size calculation with fallbacks
            max_ctx = getattr(model_config, "max_position_embeddings", None)
            if max_ctx is None:
                max_ctx = getattr(model_config, "max_sequence_length", None)
            if max_ctx is None:
                max_ctx = getattr(model_config, "sliding_window", None)
            if max_ctx is None:
                rope_scaling = getattr(model_config, "rope_scaling", None)
                if isinstance(rope_scaling, dict):
                    max_ctx = rope_scaling.get("max_position_embeddings", None)
            if max_ctx is None:
                max_ctx = 2048
            self.max_context_length = max_ctx

            self.loaded = True

            # Register in Warm Pool
            GLOBAL_WARM_POOL.add(model_name, self.model, self.tokenizer, self.processor)

            if self.on_model_loaded:
                self.on_model_loaded(model_name)

            if self.after_loading:
                self.after_loading()

    def unload(self) -> None:
        """Removes local reference and clears warm pool entry."""
        GLOBAL_WARM_POOL.remove(self.model_name)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.loaded = False

    def _apply_chat_template_cached(self, messages: List[Dict[str, Any]]) -> str:
        """Applies chat template with in-memory serialization caching."""
        msg_str = json.dumps(messages, sort_keys=True)
        if hasattr(self, "_chat_template_cache") and msg_str in self._chat_template_cache:
            return self._chat_template_cache[msg_str]

        if not hasattr(self, "_chat_template_cache"):
            self._chat_template_cache = {}

        if len(self._chat_template_cache) >= 128:
            self._chat_template_cache.pop(next(iter(self._chat_template_cache)))

        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_content = ""
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_content += c.get("text", "")
                    content = text_content
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"

        self._chat_template_cache[msg_str] = text
        return text

    def gpu_info(self) -> Dict[str, Any]:
        """Gathers properties of active GPU hardware."""
        if not torch.cuda.is_available():
            return {"status": "No CUDA device available."}
        try:
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            return {
                "device_name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "major_capability": props.major,
                "minor_capability": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
        except Exception as e:
            logger.error("Error gathering GPU info: %s", str(e))
            return {"status": "error", "message": str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Gets diagnostics detailing GPU memory utilization."""
        if not torch.cuda.is_available():
            return {"gpu_utilization_percent": "N/A", "gpu_memory_percent": "N/A"}
        try:
            device_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            nvidia_stats = get_gpu_utilization()
            
            return {
                "allocated_memory_gb": round(allocated / (1024 ** 3), 2),
                "reserved_memory_gb": round(reserved / (1024 ** 3), 2),
                "total_memory_gb": round(total_memory / (1024 ** 3), 2),
                "free_memory_gb": round((total_memory - allocated) / (1024 ** 3), 2),
                "gpu_utilization_percent": nvidia_stats.get("gpu_utilization_percent", "N/A"),
                "gpu_memory_percent": nvidia_stats.get("gpu_memory_percent", "N/A")
            }
        except Exception as e:
            logger.error("Error checking memory usage: %s", str(e))
            return {"status": "error", "message": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Gathers config settings and capabilities of active model."""
        if not self.loaded or not self.model:
            return {"status": "No model loaded."}
        try:
            cfg = self.model.config
            return {
                "model_name": self.model_name,
                "task": self.task,
                "architectures": getattr(cfg, "architectures", []),
                "model_type": getattr(cfg, "model_type", ""),
                "quantization_profile": self.quantization_profile,
                "max_context_length": self.max_context_length,
                "model_size_gb": get_model_size_gb(self.model),
                "dtype": str(self.model.dtype) if hasattr(self.model, "dtype") else "unknown"
            }
        except Exception as e:
            logger.error("Error gathering model info: %s", str(e))
            return {"status": "error", "message": str(e)}

    def get_processor_info(self) -> Dict[str, Any]:
        """Returns processor class name and tokenizer vocabulary properties."""
        if not self.loaded:
            return {"status": "No model loaded."}
        try:
            return {
                "processor_class": self.processor.__class__.__name__ if self.processor else "None",
                "tokenizer_class": self.tokenizer.__class__.__name__ if self.tokenizer else "None",
                "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
                "padding_side": self.tokenizer.padding_side if self.tokenizer else "unknown"
            }
        except Exception as e:
            logger.error("Error gathering processor info: %s", str(e))
            return {"status": "error", "message": str(e)}

    # =========================================================================
    # Helpers: Compression, Cancellation, Counters
    # =========================================================================
    def stop_generation(self) -> None:
        """Sets flag to interrupt active model generate runs."""
        self._stop_generation = True
        logger.info("Generation cancellation request flagged.")

    @staticmethod
    def compress_prompt(prompt: str) -> str:
        """Compresses whitespace, collapsible duplicate examples, and formatting to minimize token size."""
        import re
        # Collapse spaces & double newlines
        prompt = re.sub(r' +', ' ', prompt)
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)

        # Minify JSON blocks inside markdown
        def min_json(match):
            try:
                js = json.loads(match.group(1))
                return f"```json\n{json.dumps(js)}\n```"
            except Exception:
                return match.group(0)

        prompt = re.sub(r'```json\s*(\{.*?\})\s*```', min_json, prompt, flags=re.DOTALL)
        return prompt.strip()

    def count_tokens(self, text: Union[str, List[Dict[str, Any]]], images: Optional[List[Any]] = None) -> int:
        """Returns token counts for prompt text strings or message dictionaries, including image tokens."""
        if not self.tokenizer:
            return 0

        # Scan for images inside the messages content
        num_images = 0
        if isinstance(text, list):
            for msg in text:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                            num_images += 1
                elif isinstance(content, str) and "<|image_pad|>" in content:
                    num_images += content.count("<|image_pad|>")
            
            # Apply chat template
            if self.processor:
                text_str = self.processor.apply_chat_template(text, tokenize=False)
            else:
                text_str = self.tokenizer.apply_chat_template(text, tokenize=False)
        else:
            text_str = str(text)

        text_tokens = len(self.tokenizer.encode(text_str))
        
        # Add 1280 visual tokens per image (default Qwen VL representation)
        image_count = num_images
        if images:
            image_count += len(images)
            
        return text_tokens + image_count * 1280

    def prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Formats and registers prompt texts into model tensor layouts."""
        if not self.loaded or (self.tokenizer is None and self.processor is None):
            raise RuntimeError("Model parameters must be loaded before preparing inputs.")

        text = self._apply_chat_template_cached(messages)

        # Support multimodal inputs if processor is defined
        if self.processor is not None:
            proc_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images is not None and len(images) > 0:
                proc_kwargs["images"] = images
            if videos is not None and len(videos) > 0:
                proc_kwargs["videos"] = videos
            inputs = self.processor(**proc_kwargs)
        else:
            inputs = self.tokenizer([text], padding=True, return_tensors="pt")

        target_device = self.model.device if hasattr(self.model, "device") else DEVICE
        if hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        else:
            inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        return inputs

    @staticmethod
    def estimate_cost(tokens: int, mode: str = "input") -> float:
        """Returns estimated model run dollar price based on mock $0.0015 / 1K tokens."""
        rate = 0.0015 if mode == "input" else 0.0020
        return round((tokens / 1000) * rate, 5)

    def remaining_context(self, current_tokens: int) -> int:
        """Returns remaining headroom context tokens count."""
        return max(0, self.max_context_length - current_tokens)

    def _truncate_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> List[Dict[str, Any]]:
        """Iteratively drops middle dialog turns to keep conversation within context limits."""
        limit = self.max_context_length - max_new_tokens - 100  # 100 token buffer
        current_len = self.count_tokens(messages)
        if current_len <= limit:
            return messages

        logger.info("Context length exceeded (%d > %d). Truncating oldest message pairs.", current_len, limit)
        truncated = list(messages)
        has_system = truncated[0]["role"] == "system" if truncated else False

        while self.count_tokens(truncated) > limit:
            # We want to remove index 1 (the oldest user prompt) and 2 (assistant answer) to keep system prompt
            start_idx = 1 if has_system else 0
            if len(truncated) > start_idx + 1:
                # Remove two messages (turn pair)
                truncated.pop(start_idx)
                if len(truncated) > start_idx:
                    truncated.pop(start_idx)
            else:
                if len(truncated) > start_idx:
                    truncated.pop(start_idx)
                break
        return truncated

    @staticmethod
    def build_messages(
        system: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Constructs unified chat messages aggregates."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if history:
            msgs.extend(history)
        if user:
            msgs.append({"role": "user", "content": user})
        if assistant:
            msgs.append({"role": "assistant", "content": assistant})
        return msgs

    # =========================================================================
    # Inference Engine & Streaming
    # =========================================================================
    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Main inference executor managing OOM recovery, hashing, validators, and cancellation."""
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Resolve hooks
        if self.before_generation:
            self.before_generation()

        # 1. Parameter resolution
        max_new = kwargs.get("max_new_tokens", MAX_NEW_TOKENS)
        temp = kwargs.get("temperature", TEMPERATURE)
        top_p_val = kwargs.get("top_p", TOP_P)
        do_sample_val = kwargs.get("do_sample", DO_SAMPLE)
        stream = kwargs.get("stream", False)
        response_format = kwargs.get("response_format", None)
        variables = kwargs.get("variables", None)
        prompt_file = kwargs.get("prompt_file", None)
        validator = kwargs.get("validator", None)

        # 2. File and template parsing
        if prompt_file is not None:
            p_path = pathlib.Path("prompts") / prompt_file
            if not p_path.exists():
                p_path = pathlib.Path(prompt_file)
            if p_path.exists():
                prompt = p_path.read_text(encoding="utf-8")

        # White-space prompt compression
        if prompt is not None:
            prompt = self.compress_prompt(prompt)
            if variables is not None:
                prompt = format_prompt(prompt, variables)

        # 3. Assemble chat structure
        if messages is not None:
            final_messages = copy.deepcopy(messages)
        else:
            final_messages = self.build_messages(system=system_prompt, user=prompt)

        # Inject images into the last user message content if present for multimodal compatibility
        images = images or kwargs.get("images", None)
        if images:
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i]["role"] == "user":
                    content = final_messages[i]["content"]
                    if isinstance(content, str):
                        multimodal_content = []
                        for _ in images:
                            multimodal_content.append({"type": "image"})
                        multimodal_content.append({"type": "text", "text": content})
                        final_messages[i]["content"] = multimodal_content
                    elif isinstance(content, list):
                        has_image = any(isinstance(c, dict) and c.get("type") == "image" for c in content)
                        if not has_image:
                            multimodal_content = []
                            for _ in images:
                                multimodal_content.append({"type": "image"})
                            multimodal_content.extend(content)
                            final_messages[i]["content"] = multimodal_content
                    break

        # 4. Automatic Context Truncation
        final_messages = self._truncate_messages(final_messages, max_new)

        # 5. Prompt Hash Cache checking
        prompt_cache_key = self.tokenizer.apply_chat_template(final_messages, tokenize=False)
        prompt_cache_hash = hashlib.sha256(prompt_cache_key.encode('utf-8')).hexdigest()
        
        with self._cache_lock:
            if prompt_cache_hash in self._prompt_cache:
                logger.info("Prompt Hash Cache hit. Returning cached results.")
                cached_res = copy.deepcopy(self._prompt_cache[prompt_cache_hash])
                if self.after_generation:
                    self.after_generation(cached_res)
                return cached_res

        # Handle streaming path
        images = images or kwargs.get("images", None)
        videos = kwargs.get("videos", None)
        if stream:
            return self._generate_stream_yield(
                final_messages, max_new, temp, top_p_val, do_sample_val,
                images=images, videos=videos
            )

        # Build stopping cancellation criteria
        self._stop_generation = False
        def check_cancellation():
            if self._cancellation_token_fn is not None:
                if self._cancellation_token_fn():
                    return True
            return self._stop_generation
        stopping_list = StoppingCriteriaList([CancelStoppingCriteria(check_cancellation)])

        # Config builder
        gen_config = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=do_sample_val,
            temperature=temp if do_sample_val else None,
            top_p=top_p_val if do_sample_val else None,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Load inputs & model execute with OOM recovery loop
        start_time = time.perf_counter()
        max_attempts = 2
        last_error = ""

        for attempt in range(max_attempts):
            try:
                # Prepare inputs
                inputs = self.prepare_inputs(final_messages, images=images, videos=videos)
                input_length = inputs["input_ids"].shape[1]

                try:
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config,
                            stopping_criteria=stopping_list
                        )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                    is_oom = "out of memory" in str(oom_err).lower() or isinstance(oom_err, torch.cuda.OutOfMemoryError)
                    if is_oom and attempt == 0:
                        logger.warning("CUDA OOM detected inside provider. Running recovery load sequence.")
                        
                        # Evict warm pool entirely to free VRAM
                        with GLOBAL_WARM_POOL._lock:
                            GLOBAL_WARM_POOL.pool.clear()
                            GLOBAL_WARM_POOL.load_order.clear()
                            
                        self.unload()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            except Exception:
                                pass
                        gc.collect()
                        
                        self.load(self.model_name, self.task)
                        
                        inputs = self.prepare_inputs(final_messages, images=images, videos=videos)
                        input_length = inputs["input_ids"].shape[1]
                        
                        with torch.inference_mode():
                            outputs = self.model.generate(
                                **inputs,
                                generation_config=gen_config,
                                stopping_criteria=stopping_list
                            )
                    else:
                        raise oom_err

                # Support GenerateOutput wrappers
                if hasattr(outputs, "sequences"):
                    output_sequences = outputs.sequences
                elif isinstance(outputs, dict) and "sequences" in outputs:
                    output_sequences = outputs["sequences"]
                else:
                    output_sequences = outputs

                # Decoding
                generated_ids = [out_ids[input_length:] for out_ids in output_sequences]
                response_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # Cleanup variables to prevent T4 memory fragmentation
                del inputs
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Response validator checks
                parsed_json = None
                requires_json = response_format == "json" or (validator and any(k in validator for k in ["required_keys", "schema", "allow_empty_arrays", "allow_duplicate_objects"]))
                if requires_json:
                    parsed_json = extract_and_validate_json(response_text)

                is_valid, validation_error = validate_response_rules(response_text, parsed_json, validator)

                if is_valid:
                    elapsed = time.perf_counter() - start_time
                    out_tokens = len(generated_ids[0])
                    vram_usage = self.get_memory_usage()

                    result = {
                        "success": True,
                        "response": response_text,
                        "json_data": parsed_json,
                        "tokens": {
                            "input": input_length,
                            "output": out_tokens
                        },
                        "latency": round(elapsed, 4),
                        "memory": vram_usage,
                        "model": self.model_name,
                        "profile": {
                            "total_latency": round(elapsed, 4),
                            "input_tokens": input_length,
                            "output_tokens": out_tokens,
                        }
                    }

                    # Write to Prompt Hash Cache
                    with self._cache_lock:
                        if len(self._prompt_cache) >= 128:
                            self._prompt_cache.pop(next(iter(self._prompt_cache)))
                        self._prompt_cache[prompt_cache_hash] = result

                    if self.after_generation:
                        self.after_generation(result)
                    return result
                else:
                    last_error = validation_error
                    if attempt < max_attempts - 1:
                        logger.warning("Validator failure: %s. Retrying generation.", last_error)
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        gc.collect()

            except Exception as e:
                logger.error("Error running generation loop: %s", str(e))
                last_error = str(e)
                if attempt == max_attempts - 1:
                    break

        result = {
            "success": False,
            "response": "",
            "error": f"Failed to generate output: {last_error}",
            "tokens": {"input": 0, "output": 0},
            "latency": round(time.perf_counter() - start_time, 4),
            "memory": self.get_memory_usage(),
            "model": self.model_name,
            "profile": {}
        }
        if self.after_generation:
            self.after_generation(result)
        return result

    def _generate_stream_yield(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> Generator[str, None, None]:
        """Generator stream helper yielding tokens and updating streaming statistics."""
        inputs = self.prepare_inputs(messages, images=images, videos=videos)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self._stop_generation = False
        def check_cancellation():
            if self._cancellation_token_fn is not None:
                if self._cancellation_token_fn():
                    return True
            return self._stop_generation
        stopping_list = StoppingCriteriaList([CancelStoppingCriteria(check_cancellation)])

        gen_kwargs = {
            **inputs,
            "generation_config": gen_config,
            "streamer": streamer,
            "stopping_criteria": stopping_list
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        start_time = time.perf_counter()
        token_count = 0

        for new_text in streamer:
            token_count += 1
            elapsed = time.perf_counter() - start_time
            tps = token_count / elapsed if elapsed > 0 else 0.0
            
            # Update stats properties in real-time
            self.streaming_stats = {
                "tokens_per_sec": round(tps, 2),
                "elapsed": round(elapsed, 2),
                "generated_tokens": token_count,
                "remaining_estimate": round((max_new_tokens - token_count) / tps, 2) if tps > 0 else 0.0
            }
            yield new_text

        thread.join()
