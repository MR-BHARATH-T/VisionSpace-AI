"""Central Model Orchestrator (ModelManager) for VisionSpace AI.

Coordinates lazy model loading, automatic GPU memory scheduling, thread-safe request queueing,
provider routing, model lifecycle state monitoring, and 25 advanced orchestration features.
"""

from __future__ import annotations

import gc
import json
import time
import uuid
import logging
import queue
import threading
import concurrent.futures
import pathlib
import os
from typing import Any, Dict, List, Optional, Union, Generator

import torch
import config

logger = logging.getLogger("VisionSpaceAI.ModelManager")

# Config resolution
HF_MODEL = getattr(config, "MODEL", "Qwen/Qwen3-VL-8B-Instruct")
HF_VISION_MODEL = HF_MODEL
HF_TEXT_MODEL = HF_MODEL

# Model Profile mapping table
MODEL_PROFILES = {
    "vision": {
        "Fast": "Qwen/Qwen2.5-VL-3B-Instruct",
        "Balanced": HF_VISION_MODEL,
        "Accurate": "Qwen/Qwen2.5-VL-7B-Instruct"
    },
    "reasoning": {
        "Fast": "Qwen/Qwen2.5-1.5B-Instruct",
        "Balanced": HF_TEXT_MODEL,
        "Accurate": "Qwen/Qwen2.5-7B-Instruct"
    }
}

# Task dependencies
MODEL_DEPENDENCIES = {
    "spatial": "vision",
    "design": "spatial",
    "commerce": "design",
    "budget": "commerce",
    "explain": "budget"
}

# Static Capabilities Cache
MODEL_METADATA_CACHE = {
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "context_length": 32768,
        "supports_images": True,
        "supports_videos": True,
        "embedding_size": 3584,
        "generation_limit": 4096
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "context_length": 32768,
        "supports_images": False,
        "supports_videos": False,
        "embedding_size": 3584,
        "generation_limit": 4096
    },
    "default": {
        "context_length": 2048,
        "supports_images": False,
        "supports_videos": False,
        "embedding_size": 1024,
        "generation_limit": 2048
    }
}

# Central Registry of Models & Provider Mappings
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vision": {
        "model_name": HF_VISION_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "reasoning": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "spatial": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "design": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "commerce": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "budget": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "explain": {
        "model_name": HF_TEXT_MODEL,
        "provider": getattr(config, "PROVIDER", "huggingface"),
    },
    "embedding": {
        "model_name": "future-embedding",
        "provider": "future",
    },
    "ocr": {
        "model_name": "future-ocr",
        "provider": "future",
    },
    "speech": {
        "model_name": "future-speech",
        "provider": "future",
    }
}


# Processor & Tokenizer pools
class TokenizerPool:
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or "hf_cache"
        self._tokenizers: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> Any:
        with self._lock:
            if model_name not in self._tokenizers:
                from transformers import AutoTokenizer
                logger.info("Initializing Tokenizer for %s in Pool", model_name)
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=self.cache_dir, trust_remote_code=True
                )
            return self._tokenizers[model_name]


class ProcessorPool:
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or "hf_cache"
        self._processors: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> Any:
        with self._lock:
            if model_name not in self._processors:
                from transformers import AutoProcessor
                logger.info("Initializing Processor for %s in Pool", model_name)
                try:
                    self._processors[model_name] = AutoProcessor.from_pretrained(
                        model_name, cache_dir=self.cache_dir, trust_remote_code=True
                    )
                except Exception:
                    self._processors[model_name] = None
            return self._processors[model_name]


# Subscriber event system
class EventSystem:
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Any]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Any) -> None:
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)

    def emit(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            callbacks = list(self._listeners.get(event_type, []))
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning("Error in event listener for '%s': %s", event_type, str(e))


class Job:
    def __init__(self, request_id: str, priority_val: int, task: str, prompt: Optional[str], messages: Optional[List[Dict[str, Any]]], kwargs: Dict[str, Any]) -> None:
        self.request_id = request_id
        self.priority_val = priority_val  # 0=HIGH, 1=NORMAL, 2=LOW
        self.task = task
        self.prompt = prompt
        self.messages = messages
        self.kwargs = kwargs
        self.event = threading.Event()
        self.result: Any = None
        self.exception: Optional[Exception] = None
        self.cancelled = False


class ModelManager:
    """Singleton Model Manager class orchestrating all models and execution contexts."""

    _instance: Optional[ModelManager] = None
    _lock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> ModelManager:
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cache_dir = getattr(config, "HF_CACHE_DIR", "hf_cache")
        self.providers: Dict[str, Any] = {}

        # Tokenizer and Processor pools
        self.tokenizer_pool = TokenizerPool(self.cache_dir)
        self.processor_pool = ProcessorPool(self.cache_dir)

        # Event system
        self.events = EventSystem()

        # Thread Locks
        self._init_lock = threading.RLock()
        self._generate_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._queue_lock = threading.RLock()
        self._counter_lock = threading.Lock()

        # Model Lifecycles states
        self._lifecycle_states: Dict[str, str] = {
            task: "Cold" for task in MODEL_REGISTRY
        }

        # Last used timestamps for auto cleanup
        self._last_used_time: Dict[str, float] = {
            task: time.time() for task in MODEL_REGISTRY
        }

        # Session metrics tracking
        self.session_id = str(uuid.uuid4())
        self.prompt_count = 0
        self.conversation_history: List[Dict[str, Any]] = []

        # Request management
        self._request_counter = 0
        self._pending_jobs: List[Job] = []
        self._running_jobs: Dict[str, Job] = {}
        self._cancelled_requests: Set[str] = set()

        # Thread Pool Executor for Async calls
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        # Batch Scheduler queues and workers
        self._batch_queues: Dict[str, queue.Queue] = {}
        self._batch_workers: Dict[str, threading.Thread] = {}
        self._batch_results: Dict[str, Dict[str, Any]] = {}
        self._batch_locks: Dict[str, threading.RLock] = {}

        # Analytics tracking
        self.analytics = {
            "calls": {task: 0 for task in MODEL_REGISTRY},
            "input_tokens": {task: 0 for task in MODEL_REGISTRY},
            "output_tokens": {task: 0 for task in MODEL_REGISTRY},
            "latency": {task: 0.0 for task in MODEL_REGISTRY},
            "oom_count": {task: 0 for task in MODEL_REGISTRY}
        }

        self.stats = {
            "load_times": {},
            "inference_times": [],
            "cache_hits": 0,
            "oom_count": 0
        }

        # Start background clean-up thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_timer_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("ModelManager initialized successfully. Session ID: %s", self.session_id)

    # =========================================================================
    # Provider Loader Delegation
    # =========================================================================
    def _get_provider(self, provider_name: str, task: Optional[str] = None) -> Any:
        """Retrieves or instantiates provider implementation wrapper."""
        resolved_task = task or 'reasoning'
        meta = MODEL_REGISTRY.get(resolved_task, {})
        model_name = meta.get("model_name")
        
        # Share provider if task uses HF_VISION_MODEL or HF_TEXT_MODEL
        if provider_name == "huggingface":
            if model_name == HF_VISION_MODEL:
                resolved_task = "vision"
            elif model_name == HF_TEXT_MODEL:
                resolved_task = "reasoning"
            
        key = f"{provider_name}:{resolved_task}"
        if key not in self.providers:
            with self._init_lock:
                if key not in self.providers:
                    if provider_name == "huggingface":
                        from providers.huggingface_provider import HuggingFaceProvider
                        self.providers[key] = HuggingFaceProvider()
                    elif provider_name == "deepinfra":
                        from providers.deepinfra_provider import DeepInfraProvider
                        self.providers[key] = DeepInfraProvider()
                    else:
                        raise ValueError(f"Provider '{provider_name}' is currently unsupported.")
        return self.providers[key]

    # =========================================================================
    # Context Budget & Token utilities
    # =========================================================================
    def count_tokens(self, task: str, text: Union[str, List[Dict[str, Any]]]) -> int:
        """Returns the token counts for text or message list using pooled tokenizer."""
        meta = MODEL_REGISTRY.get(task, MODEL_REGISTRY["reasoning"])
        model_name = meta["model_name"]
        tokenizer = self.tokenizer_pool.get(model_name)
        if isinstance(text, list):
            # Fallback representation of dict message text
            flat_text = ""
            for m in text:
                flat_text += f"{m.get('role', '')}: {m.get('content', '')}\n"
            return len(tokenizer.encode(flat_text))
        return len(tokenizer.encode(text))

    def max_context(self, task: str) -> int:
        """Returns max context length allowed by model metadata cache."""
        meta = MODEL_REGISTRY.get(task, MODEL_REGISTRY["reasoning"])
        model_name = meta["model_name"]
        entry = MODEL_METADATA_CACHE.get(model_name, MODEL_METADATA_CACHE["default"])
        return entry.get("context_length", 2048)

    def remaining_context(self, task: str, current_tokens: int) -> int:
        """Returns context window headroom remaining."""
        return max(0, self.max_context(task) - current_tokens)

    # =========================================================================
    # Environment version manager
    # =========================================================================
    def environment(self) -> Dict[str, str]:
        """Exposes diagnostics detailing current package version metadata."""
        import sys
        import transformers
        try:
            import accelerate
            acc_version = accelerate.__version__
        except ImportError:
            acc_version = "not installed"
            
        try:
            import bitsandbytes
            bnb = bitsandbytes.__version__
        except ImportError:
            bnb = "not installed"

        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "transformers": transformers.__version__,
            "accelerate": acc_version,
            "torch": torch.__version__,
            "bitsandbytes": bnb,
            "cuda": cuda_ver,
            "device": dev,
            "python": sys.version.split()[0]
        }

    # =========================================================================
    # Model Metadata Capabilities
    # =========================================================================
    def get_model_metadata(self, task: str) -> Dict[str, Any]:
        """Provides capabilities config mapping for the task model."""
        meta = MODEL_REGISTRY.get(task, MODEL_REGISTRY["reasoning"])
        model_name = meta["model_name"]
        return MODEL_METADATA_CACHE.get(model_name, MODEL_METADATA_CACHE["default"])

    # =========================================================================
    # VRAM Memory Scheduler & Switching
    # =========================================================================
    def _schedule_vram(self, target_task: str) -> None:
        """Kaggle-friendly memory scheduler.

        Aggressively evicts other models to free VRAM before loading next.
        """
        if not torch.cuda.is_available():
            return

        try:
            target_meta = MODEL_REGISTRY.get(target_task, {})
            target_model_name = target_meta.get("model_name")

            # Aggressively unload any loaded model utilizing different weights
            # to prevent Hugging Face device_map auto offloading traps
            for task in list(MODEL_REGISTRY.keys()):
                if task != target_task and self.is_loaded(task):
                    meta = MODEL_REGISTRY[task]
                    if meta.get("model_name") != target_model_name:
                        logger.warning("VRAM Scheduler: Preemptively unloading different model for task '%s' to prevent OOM.", task)
                        self.unload_model(task)

            # Force device cleanup
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            logger.warning("Error in VRAM memory scheduler: %s", str(e))

    # =========================================================================
    # Lifecycles
    # =========================================================================
    def load_model(self, task: str, profile: Optional[str] = None) -> bool:
        """Lazy model loader wrapper coordinating states and memory thresholds."""
        if task not in MODEL_REGISTRY:
            raise ValueError(f"Task '{task}' is not registered in MODEL_REGISTRY.")

        meta = MODEL_REGISTRY[task]
        provider_name = meta["provider"]
        if provider_name != "future":
            provider_name = getattr(config, "PROVIDER", "huggingface")

        if provider_name == "future":
            logger.warning("Task '%s' maps to a future provider model. Skipping load.", task)
            return False

        # Resolve model name dynamically based on the active provider
        if provider_name == "deepinfra":
            model_name = getattr(config, "MODEL", "deepinfra/alibaba/qwen3-vl-instruct")
        else:
            model_name = meta["model_name"]
            if profile and task in MODEL_PROFILES and profile in MODEL_PROFILES[task]:
                model_name = MODEL_PROFILES[task][profile]
                logger.info("Dynamic Profile '%s' selected model: %s", profile, model_name)

        provider = self._get_provider(provider_name, task)

        # If already loaded
        if provider.loaded and provider.model_name == model_name:
            self._set_state(task, "Ready")
            self._last_used_time[task] = time.time()
            return True

        with self._init_lock:
            # Recheck status inside lock
            if provider.loaded and provider.model_name == model_name:
                return True

            logger.info("Requesting model load for task '%s' (%s) via provider '%s'", task, model_name, provider_name)
            self._set_state(task, "Loading")
            self.events.emit("model_loading", task=task, model_name=model_name)

            if provider_name == "huggingface":
                # Download Manager Check
                self._verify_and_download_model(model_name)

                # Model Integrity Check
                self._verify_model_integrity(model_name)

                # Execute scheduling checks
                self._schedule_vram(task)

            start_time = time.perf_counter()
            try:
                if provider_name == "huggingface":
                    # Fetch tokenizer and processor from central pools and configure provider
                    provider.tokenizer = self.tokenizer_pool.get(model_name)
                    provider.processor = self.processor_pool.get(model_name)

                # Load provider weights
                provider.load(model_name, task)

                self._set_state(task, "Ready")
                self._last_used_time[task] = time.time()

                load_duration = time.perf_counter() - start_time
                self.stats["load_times"][task] = round(load_duration, 3)
                logger.info("Successfully loaded task '%s' in %.3f seconds.", task, load_duration)
                self.events.emit("model_loaded", task=task, model_name=model_name)

                # Automatic Warmup Pass
                self._run_warmup_pass(provider, model_name)

                return True
            except Exception as e:
                self._set_state(task, "Error")
                logger.error("Failed to load model for task '%s': %s", task, str(e))
                self.events.emit("model_load_error", task=task, model_name=model_name, error=str(e))
                return False

    def unload_model(self, task: str) -> None:
        """Removes model reference and triggers memory garbage collection."""
        if not self.is_loaded(task):
            self._set_state(task, "Cold")
            return

        with self._init_lock:
            if self.is_loaded(task):
                logger.info("Unloading model for task '%s'", task)
                self._set_state(task, "Unloading")
                
                meta = MODEL_REGISTRY[task]
                provider = self._get_provider(meta["provider"], task)
                provider.unload()

                self._set_state(task, "Cold")
                self.events.emit("model_unloaded", task=task, model_name=meta["model_name"])

                # Cleanup device caches
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                gc.collect()
                logger.info("Task '%s' unloaded successfully.", task)

    # =========================================================================
    # Generation & Streaming API
    # =========================================================================
    def generate(
        self,
        task: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        priority: str = "NORMAL",
        profile: Optional[str] = None,
        response_format: Optional[str] = None,
        validator: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Main synchronized inference interface orchestrating lazy switching and queues."""
        if task not in MODEL_REGISTRY:
            raise ValueError(f"Task '{task}' is not registered.")

        # Resolve Model dependencies first (e.g. running vision prior to spatial)
        messages = self._resolve_dependencies(task, prompt, messages, priority=priority, profile=profile, **kwargs)

        # Generate unique request id
        req_id = self._next_request_id(task)
        self.events.emit("inference_started", task=task, request_id=req_id)

        # Wrap query parameters inside a Job
        priority_map = {"HIGH": 0, "NORMAL": 1, "LOW": 2}
        pri_val = priority_map.get(priority.upper(), 1)
        job = Job(req_id, pri_val, task, prompt, messages, kwargs)

        with self._queue_lock:
            self._pending_jobs.append(job)
            self._pending_jobs.sort(key=lambda j: j.priority_val)

        # Execute loop coordinating priority execution serialization
        while not job.event.is_set():
            if req_id in self._cancelled_requests:
                job.cancelled = True
                job.exception = RuntimeError(f"Request {req_id} was cancelled by user.")
                job.event.set()
                break

            with self._generate_lock:
                next_job = None
                with self._queue_lock:
                    if self._pending_jobs and self._pending_jobs[0].request_id == req_id:
                        next_job = self._pending_jobs.pop(0)

                if next_job:
                    try:
                        self._running_jobs[req_id] = next_job
                        next_job.result = self._execute_generation(
                            task=next_job.task,
                            prompt=next_job.prompt,
                            messages=next_job.messages,
                            system_prompt=system_prompt,
                            profile=profile,
                            response_format=response_format,
                            validator=validator,
                            request_id=req_id,
                            **next_job.kwargs
                        )
                    except Exception as e:
                        next_job.exception = e
                    finally:
                        self._running_jobs.pop(req_id, None)
                        next_job.event.set()
                    break
            time.sleep(0.02)

        if job.exception:
            self.events.emit("inference_error", task=task, request_id=req_id, error=str(job.exception))
            raise job.exception

        self.events.emit("inference_finished", task=task, request_id=req_id, result=job.result)
        return job.result

    def _execute_generation(
        self,
        task: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
        profile: Optional[str],
        response_format: Optional[str],
        validator: Optional[Dict[str, Any]],
        request_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Performs actual model lazy switching, OOM retries, and validator schemas checks."""
        self._last_used_time[task] = time.time()
        
        # 1. Ensure target model is loaded (supporting dynamic profiling sizes)
        if not self.load_model(task, profile=profile):
            raise RuntimeError(f"Failed to load model weights for task '{task}'. See logs for details.")
        self._set_state(task, "Generating")
        self.prompt_count += 1
        
        meta = MODEL_REGISTRY[task]
        provider = self._get_provider(meta["provider"], task)

        # Centralized Context Budget Trim/Prune step
        if messages:
            messages = self._apply_context_budget(task, messages, kwargs.get("max_new_tokens", 2048))

        # Check and inject Cancellation Stopping Criteria if supported
        if hasattr(provider, "set_cancellation_token"):
            provider.set_cancellation_token(lambda: request_id in self._cancelled_requests)

        start_time = time.perf_counter()
        try:
            # 2. Execute inference supporting automatic OOM Retry strategy
            result = self._execute_inference_with_retry(provider, task, prompt, messages, system_prompt, **kwargs)
            
            # Format and standard output fields mapping
            if isinstance(result, Generator):
                text_parts = []
                for chunk in result:
                    text_parts.append(chunk)
                response_text = "".join(text_parts)
                result = {
                    "success": True,
                    "response": response_text,
                    "tokens": {"input": 0, "output": 0},
                    "latency": round(time.perf_counter() - start_time, 4),
                    "memory": self.get_memory_usage(),
                    "model": meta["model_name"],
                    "profile": {}
                }

            # Response JSON validation and Self-Correction retry loop
            if response_format == "json" or validator:
                is_valid, validated_obj, err = self._validate_response_schema(result.get("response", ""), validator)
                if not is_valid:
                    logger.warning("Response validation failed: %s. Retrying with self-correction prompt.", err)
                    correction_prompt = f"Error: output did not match requirements: {err}. Please correct the output and return a valid JSON payload."
                    
                    # Create messages chain with assistant output and validation error
                    correction_history = []
                    if messages:
                        correction_history = list(messages)
                    elif prompt:
                        correction_history = [{"role": "user", "content": prompt}]
                    
                    correction_history.append({"role": "assistant", "content": result.get("response", "")})
                    correction_history.append({"role": "user", "content": correction_prompt})
                    
                    result = self._execute_inference_with_retry(provider, task, None, correction_history, system_prompt, **kwargs)
                    # final validation check
                    is_valid, validated_obj, err = self._validate_response_schema(result.get("response", ""), validator)
                    result["validation_success"] = is_valid
                    result["validation_error"] = err
                else:
                    result["validation_success"] = True
                    result["validation_error"] = None

            # Analytics tracking update
            self.analytics["calls"][task] += 1
            latency = result.get("latency", 0.0)
            self.analytics["latency"][task] = (self.analytics["latency"][task] * 0.8) + (latency * 0.2)
            self.stats["inference_times"].append(latency)
            
            self._set_state(task, "Idle")
            return result

        except Exception as e:
            self._set_state(task, "Error")
            import traceback
            traceback.print_exc()
            logger.error("Inference failure during generate: %s", str(e))
            return {
                "success": False,
                "response": "",
                "error": str(e),
                "tokens": {"input": 0, "output": 0},
                "latency": round(time.perf_counter() - start_time, 4),
                "memory": self.get_memory_usage(),
                "model": meta["model_name"],
                "profile": {}
            }

    def _execute_inference_with_retry(self, provider: Any, task: str, prompt: Optional[str], messages: Optional[List[Dict[str, Any]]], system_prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
        """Tiers query execution to handle OOM errors gracefully."""
        try:
            return provider.generate(prompt=prompt, messages=messages, system_prompt=system_prompt, **kwargs)
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower() or isinstance(e, RuntimeError):
                logger.warning("CUDA OOM detected inside provider. Initiating recovery cleanup and retry.")
                self.analytics["oom_count"][task] += 1
                self.stats["oom_count"] += 1
                
                # Tier 1 Recovery: Free device VRAM cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    return provider.generate(prompt=prompt, messages=messages, system_prompt=system_prompt, **kwargs)
                except Exception as retry_err:
                    # Tier 2 Recovery: Evict other models
                    logger.warning("Tier 1 recovery failed. Attempting Tier 2 (Evicting other models).")
                    for active_task in list(MODEL_REGISTRY.keys()):
                        if active_task != task and self.is_loaded(active_task):
                            self.unload_model(active_task)
                    
                    try:
                        return provider.generate(prompt=prompt, messages=messages, system_prompt=system_prompt, **kwargs)
                    except Exception as last_err:
                        logger.error("All generation retry tiers failed. Raising OOM exception.")
                        raise last_err
            else:
                raise e

    def generate_async(self, task: str, prompt: Optional[str] = None, **kwargs: Any) -> concurrent.futures.Future:
        """Asynchronous execution mapping. Returns Future reference."""
        return self._executor.submit(self.generate, task, prompt, **kwargs)

    def generate_stream(
        self,
        task: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Any] = None,
        on_complete: Optional[Any] = None,
        on_error: Optional[Any] = None,
        on_cancel: Optional[Any] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """Exposes token generator supporting Gradio callbacks and active cancellation."""
        if task not in MODEL_REGISTRY:
            raise ValueError(f"Task '{task}' is not registered.")

        # Ensure model is loaded
        self.load_model(task)
        self._set_state(task, "Generating")
        self._last_used_time[task] = time.time()
        
        meta = MODEL_REGISTRY[task]
        provider = self._get_provider(meta["provider"], task)
        kwargs["stream"] = True

        req_id = self._next_request_id(task)
        self.events.emit("inference_started", task=task, request_id=req_id)

        if hasattr(provider, "set_cancellation_token"):
            provider.set_cancellation_token(lambda: req_id in self._cancelled_requests)

        try:
            generator = provider.generate(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )
            full_response = []
            for token in generator:
                if req_id in self._cancelled_requests:
                    logger.info("Streaming generation request %s was cancelled by user.", req_id)
                    if on_cancel:
                        on_cancel()
                    break
                full_response.append(token)
                if on_token:
                    on_token(token)
                yield token
            
            response_text = "".join(full_response)
            if on_complete:
                on_complete(response_text)
            self._set_state(task, "Idle")
        except Exception as e:
            self._set_state(task, "Error")
            logger.error("Error yielding streaming tokens: %s", str(e))
            if on_error:
                on_error(e)
            yield f"ERROR: {str(e)}"

    def generate_batch(
        self,
        task: str,
        prompts: List[str],
        **kwargs: Any
    ) -> List[str]:
        """Queues batch tasks via ModelManager's worker queue loop."""
        if not prompts:
            return []
            
        self.load_model(task)
        self._last_used_time[task] = time.time()
        
        # Lazy initialize batch scheduler state for this task
        with self._init_lock:
            if task not in self._batch_queues:
                self._batch_queues[task] = queue.Queue()
                self._batch_results[task] = {}
                self._batch_locks[task] = threading.RLock()
                
                # Start background thread worker loop
                def worker_loop(t=task):
                    while True:
                        try:
                            task_id, prompt, batch_kwargs = self._batch_queues[t].get()
                            try:
                                res = self.generate(task=t, prompt=prompt, **batch_kwargs)
                                self._batch_results[t][task_id] = res
                            except Exception as e:
                                self._batch_results[t][task_id] = {"success": False, "error": str(e)}
                            finally:
                                self._batch_queues[t].task_done()
                        except Exception as e:
                            logger.error("Error in batch worker for task %s: %s", t, str(e))
                            time.sleep(1.0)
                            
                t_thread = threading.Thread(target=worker_loop, daemon=True)
                t_thread.start()
                self._batch_workers[task] = t_thread

        task_ids = []
        for prompt in prompts:
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)
            self._batch_queues[task].put((task_id, prompt, kwargs))

        # Wait for all tasks to complete
        results = {}
        while len(results) < len(prompts):
            with self._batch_locks[task]:
                for tid in task_ids:
                    if tid in self._batch_results[task] and tid not in results:
                        results[tid] = self._batch_results[task][tid]
            time.sleep(0.05)

        # Map back to ordered responses
        ordered = []
        for tid in task_ids:
            res = results[tid]
            ordered.append(res["response"] if res.get("success", False) else f"ERROR: {res.get('error')}")
        return ordered

    def cancel_request(self, request_id: str) -> None:
        """Interrupts and stops active running generations matching request ID."""
        logger.info("Cancellation requested for ID: %s", request_id)
        self._cancelled_requests.add(request_id)
        self.events.emit("inference_cancelled", request_id=request_id)

    # =========================================================================
    # Internals & Verification helpers
    # =========================================================================
    def _next_request_id(self, task: str) -> str:
        with self._counter_lock:
            self._request_counter += 1
            count = self._request_counter
        return f"{task.upper()}-{count:06d}"

    def _resolve_dependencies(self, task: str, prompt: Optional[str], messages: Optional[List[Dict[str, Any]]], **kwargs: Any) -> List[Dict[str, Any]]:
        """Resolves model prerequisites, automatically executing dependee models recursively."""
        dep = MODEL_DEPENDENCIES.get(task)
        if not dep:
            return messages or []

        # Resolve parent dependencies first
        messages = self._resolve_dependencies(dep, prompt, messages, **kwargs)

        # Calculate recursive depth of target task in graph
        depth = 0
        curr = task
        while curr in MODEL_DEPENDENCIES:
            depth += 1
            curr = MODEL_DEPENDENCIES[curr]

        # Check if dependee task's output is already in messages history
        has_dep_output = False
        if messages:
            assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
            if len(assistant_msgs) >= depth:
                has_dep_output = True

        if not has_dep_output:
            logger.info("Task dependency '%s' is missing. Running dependee generation first.", dep)
            dep_res = self.generate(task=dep, prompt=prompt, messages=messages, **kwargs)
            if messages is None:
                messages = []
            messages.append({"role": "assistant", "content": dep_res.get("response", "")})

        return messages

    def _apply_context_budget(self, task: str, messages: List[Dict[str, Any]], expected_new_tokens: int) -> List[Dict[str, Any]]:
        """Prunes historical conversation turns to fit within context length budget."""
        max_ctx = self.max_context(task)
        token_count = self.count_tokens(task, messages)
        
        # If exceeding headroom, summarize/prune oldest turns
        if (token_count + expected_new_tokens) >= max_ctx:
            logger.warning("Headroom budget exceeded (%d / %d tokens). Performing conversation history pruning.", token_count, max_ctx)
            # Retain system prompt (index 0) and drop oldest user/assistant turns
            system_prompt = None
            start_index = 0
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0]
                start_index = 1
            
            history_turns = messages[start_index:]
            # Summarize or prune older messages (remove first user-assistant pair)
            if len(history_turns) >= 2:
                pruned = history_turns[2:]
                messages = [system_prompt] + pruned if system_prompt else pruned
                # Recheck recursively
                return self._apply_context_budget(task, messages, expected_new_tokens)
        return messages

    def _validate_response_schema(self, text: str, validator: Optional[Dict[str, Any]]) -> tuple[bool, Any, Optional[str]]:
        """Parses output text to JSON object and validates against schema restrictions."""
        try:
            from utils.json_parser import extract_and_validate_json, validate_response_rules
            # Clean boundaries markup
            parsed_obj = extract_and_validate_json(text)
            if validator:
                is_valid, err = validate_response_rules(text, parsed_obj, validator)
                if not is_valid:
                    return False, None, err
            return True, parsed_obj, None
        except Exception as e:
            return False, None, str(e)

    def _verify_and_download_model(self, model_name: str) -> None:
        """Verifies cache directory presence and downloads model weights using HuggingFace Hub."""
        local_path = os.path.join(self.cache_dir, "models--" + model_name.replace("/", "--"))
        if not os.path.exists(local_path):
            logger.info("Model '%s' not found locally. Initiating Hugging Face download.", model_name)
            self.events.emit("download_started", model_name=model_name)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_name,
                    cache_dir=self.cache_dir,
                    token=getattr(config, "HF_TOKEN", None),
                    ignore_patterns=["*.msgpack", "*.h5"]
                )
                self.events.emit("download_finished", model_name=model_name)
            except Exception as e:
                logger.error("Download failed for model %s: %s", model_name, str(e))
                self.events.emit("download_error", model_name=model_name, error=str(e))
                raise e

    def _verify_model_integrity(self, model_name: str) -> None:
        """Scans cached model directories, checking completeness and repairing if corrupted."""
        # Simple local files check
        local_path = os.path.join(self.cache_dir, "models--" + model_name.replace("/", "--"))
        if os.path.exists(local_path):
            config_file_found = False
            for root, dirs, files in os.walk(local_path):
                if "config.json" in files:
                    config_file_found = True
                    # check config.json size is > 0
                    config_path = os.path.join(root, "config.json")
                    if os.path.getsize(config_path) == 0:
                        logger.warning("Corrupted config.json (0 bytes) found. Triggering repair download.")
                        self._verify_and_download_model(model_name)
                        break
            if not config_file_found:
                logger.warning("config.json missing in local cache repository. Triggering repair download.")
                self._verify_and_download_model(model_name)

    def _run_warmup_pass(self, provider: Any, model_name: str) -> None:
        """Executes a tiny dummy prompt to warm CUDA streams and allocate initial caches."""
        try:
            logger.info("Running automatic model warmup pass for: %s", model_name)
            dummy_messages = [{"role": "user", "content": "Warmup query"}]
            # Disable streaming/callbacks and limit new tokens to 1
            provider.generate(messages=dummy_messages, max_new_tokens=1)
            logger.info("Model warmup pass completed successfully.")
        except Exception as e:
            logger.warning("Model warmup pass raised warning/error: %s", str(e))

    def _cleanup_timer_loop(self) -> None:
        """Daemon thread loop unloading models left idle for 15 minutes."""
        while True:
            time.sleep(60)
            now = time.time()
            with self._init_lock:
                for task, last_used in list(self._last_used_time.items()):
                    if self.is_loaded(task):
                        # 15 minutes = 900 seconds
                        if (now - last_used) > 900:
                            logger.info("Model for task '%s' has been idle for 15 minutes. Unloading automatically.", task)
                            self.unload_model(task)

    # =========================================================================
    # Health Monitoring & Status
    # =========================================================================
    def is_loaded(self, task: str) -> bool:
        """Checks if model weights for task are active in VRAM."""
        if task not in MODEL_REGISTRY:
            return False
        provider_name = meta["provider"]
        if provider_name != "future":
            provider_name = getattr(config, "PROVIDER", "huggingface")
        else:
            return False
        provider = self._get_provider(provider_name, task)
        return provider.loaded

    def is_ready(self, task: str) -> bool:
        """Checks if lifecycle state is Ready or Idle."""
        state = self.status(task)
        return state in ["Ready", "Idle"]

    def status(self, task: str) -> str:
        """Returns active lifecycle status string for task."""
        with self._state_lock:
            return self._lifecycle_states.get(task, "Cold")

    def ping(self) -> str:
        """Heartbeat check."""
        return "pong"

    def health_check(self, task: str = "reasoning") -> bool:
        """Executes dummy query validation checks to test stack soundness."""
        try:
            self.load_model(task)
            meta = MODEL_REGISTRY[task]
            provider = self._get_provider(meta["provider"], task)
            return provider.health_check() if hasattr(provider, "health_check") else True
        except Exception as e:
            logger.error("Health check failed for task '%s': %s", task, str(e))
            return False

    def get_model_info(self, task: str) -> Dict[str, Any]:
        """Exposes detailed dictionary about loaded task models config properties."""
        if not self.is_loaded(task):
            return {"status": "Model not loaded", "task": task}
        meta = MODEL_REGISTRY[task]
        provider = self._get_provider(meta["provider"], task)
        return provider.get_model_info()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Provides CPU and GPU diagnostics."""
        active_provider = self._get_provider("huggingface", "reasoning")
        return active_provider.get_memory_usage()

    def get_analytics_report(self) -> Dict[str, Any]:
        """Exposes diagnostic summary of model invocation analytics."""
        return self.analytics

    # =========================================================================
    # Cache management forwarding
    # =========================================================================
    def clear_cache(self) -> bool:
        """Safely cleans cache repository directories."""
        active_provider = self._get_provider("huggingface", "reasoning")
        return active_provider.clear_cache()

    def cache_size(self) -> float:
        """Gets cache occupied storage size in GB."""
        active_provider = self._get_provider("huggingface", "reasoning")
        return active_provider.cache_size()

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists cached repository objects."""
        active_provider = self._get_provider("huggingface", "reasoning")
        return active_provider.list_cached_models()

    # =========================================================================
    # Lifecycle & Cleanup Hooks
    # =========================================================================
    def shutdown(self) -> None:
        """Performs clean orchestrator shutdown unloading all loaded instances."""
        logger.info("Initiating model manager shutdown procedures.")
        for task in list(MODEL_REGISTRY.keys()):
            try:
                if self.is_loaded(task):
                    self.unload_model(task)
            except Exception as e:
                logger.error("Error unloading task '%s' during shutdown: %s", task, str(e))
        self.providers.clear()
        logger.info("Shutdown completed successfully.")

    # =========================================================================
    # Internals
    # =========================================================================
    def _set_state(self, task: str, state: str) -> None:
        with self._state_lock:
            self._lifecycle_states[task] = state
            logger.debug("Task '%s' state transitioned to: %s", task, state)

    def _track_session_history(self, prompt: Optional[str], messages: Optional[List[Dict[str, Any]]], result: Dict[str, Any]) -> None:
        """Track history metrics per session."""
        try:
            content = prompt if prompt else str(messages)
            entry = {
                "timestamp": time.time(),
                "prompt_snippet": content[:100] if content else "",
                "success": result.get("success", False),
                "latency": result.get("latency", 0.0),
                "tokens": result.get("tokens", {"input": 0, "output": 0})
            }
            self.conversation_history.append(entry)
            if len(self.conversation_history) > 100:
                self.conversation_history.pop(0)
        except Exception:
            pass
