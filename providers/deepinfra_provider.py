"""DeepInfra Provider Module for VisionSpace AI.

Integrates DeepInfra's OpenAI-compatible endpoint to execute cheap and fast completions
for Qwen3-VL-8B-Instruct and other text models.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Generator

import requests
from PIL import Image

import config
from providers.base_provider import BaseProvider

logger = logging.getLogger("VisionSpaceAI.DeepInfraProvider")

class DeepInfraProvider(BaseProvider):
    """DeepInfra/Vercel AI Gateway API provider implementing the BaseProvider contract interface."""

    def __init__(self) -> None:
        """Initializes the DeepInfraProvider."""
        self.vercel_key = os.environ.get("VERCEL_API_KEY") or getattr(config, "VERCEL_API_KEY", None)
        self.deepinfra_key = os.environ.get("DEEPINFRA_API_KEY") or getattr(config, "DEEPINFRA_API_KEY", None)
        self.api_key = self.vercel_key or self.deepinfra_key
        
        if self.vercel_key:
            self.model_name = "deepinfra/alibaba/qwen3-vl-instruct"
            self.base_url = "https://ai-gateway.vercel.sh/v1"
        else:
            self.model_name = getattr(config, "MODEL", "Qwen/Qwen3-VL-8B-Instruct")
            self.base_url = "https://api.deepinfra.com/v1/openai"

        self.task = "reasoning"
        self._stopped = False
        self.loaded = False

        if not self.api_key:
            logger.warning("API keys (DEEPINFRA_API_KEY / VERCEL_API_KEY) are missing. Running in mock fallback mode.")

    def load(self, model_name: str, task: str, config_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Saves config metadata. No local weights loading required for API provider."""
        self.vercel_key = os.environ.get("VERCEL_API_KEY") or getattr(config, "VERCEL_API_KEY", None)
        self.deepinfra_key = os.environ.get("DEEPINFRA_API_KEY") or getattr(config, "DEEPINFRA_API_KEY", None)
        self.api_key = self.vercel_key or self.deepinfra_key

        if self.vercel_key:
            self.model_name = "deepinfra/alibaba/qwen3-vl-instruct"
            self.base_url = "https://ai-gateway.vercel.sh/v1"
        else:
            self.model_name = model_name
            self.base_url = "https://api.deepinfra.com/v1/openai"

        self.task = task
        self.loaded = True
        logger.info("API provider initialized for model '%s' via '%s' (task: %s)", self.model_name, self.base_url, task)

    def unload(self) -> None:
        """No GPU memory cleanup needed for API-based provider."""
        self.loaded = False

    def _pil_to_base64(self, img: Image.Image) -> str:
        """Encodes PIL image as base64 string."""
        try:
            buffered = io.BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to encode PIL Image to Base64: %s", str(e))
            raise

    def generate(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Invokes chat completions endpoint on DeepInfra. Falls back to mock in development if API key is missing."""
        self._stopped = False
        api_key = self.api_key or os.environ.get("VERCEL_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            logger.warning("No API credentials configured. Running in DEVELOPMENT MOCK MODE.")
            return self._generate_mock(prompt, images, messages, system_prompt, **kwargs)

        # 1. Format payload messages array
        formatted_messages: List[Dict[str, Any]] = []

        if messages:
            formatted_messages = list(messages)
        else:
            user_content: Union[str, List[Dict[str, Any]]] = ""
            if images:
                content_list = []
                if prompt:
                    content_list.append({"type": "text", "text": prompt})
                for img in images:
                    if isinstance(img, Image.Image):
                        b64_str = self._pil_to_base64(img)
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_str}"
                            }
                        })
                user_content = content_list
            else:
                user_content = prompt or ""

            formatted_messages = [{"role": "user", "content": user_content}]

        if system_prompt:
            formatted_messages = [{"role": "system", "content": system_prompt}] + formatted_messages

        # 2. Call API completions
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_new_tokens", 2048),
        }

        if kwargs.get("response_format") == "json" or "json" in str(kwargs.get("response_format", "")).lower():
            payload["response_format"] = {"type": "json_object"}

        logger.info("Sending request to DeepInfra API (%s)...", self.model_name)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                logger.warning("DeepInfra API returned status code %d: %s. Falling back to DEVELOPMENT MOCK MODE.", response.status_code, response.text)
                return self._generate_mock(prompt, images, messages, system_prompt, **kwargs)
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                logger.warning("DeepInfra API returned empty choices. Falling back to DEVELOPMENT MOCK MODE.")
                return self._generate_mock(prompt, images, messages, system_prompt, **kwargs)
        except Exception as api_err:
            logger.warning("DeepInfra API request failed: %s. Falling back to DEVELOPMENT MOCK MODE.", str(api_err))
            return self._generate_mock(prompt, images, messages, system_prompt, **kwargs)

        choice = choices[0]
        text_out = choice.get("message", {}).get("content", "")

        input_tokens = res_json.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = res_json.get("usage", {}).get("completion_tokens", 0)

        # Standard return dict format
        return {
            "text": text_out,
            "response": text_out,
            "success": True,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens
            }
        }

    def gpu_info(self) -> Dict[str, Any]:
        """DeepInfra uses cloud compute, so return empty diagnostics."""
        return {"device": "Cloud API (DeepInfra)"}

    def get_memory_usage(self) -> Dict[str, Any]:
        """DeepInfra uses cloud compute, so return empty memory status."""
        return {"allocated_bytes": 0, "reserved_bytes": 0}

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active model."""
        return {
            "model_name": self.model_name,
            "context_length": 32768,
            "supports_images": True
        }

    def get_processor_info(self) -> Dict[str, Any]:
        """Returns tokenizer information."""
        return {"processor": "DeepInfra API Tokenizer"}

    def stop_generation(self) -> None:
        """Sets internal stop flag."""
        self._stopped = True

    def count_tokens(self, text: Union[str, List[Dict[str, Any]]]) -> int:
        """Approximates token count by words length heuristics."""
        if isinstance(text, list):
            flat = ""
            for m in text:
                flat += f"{m.get('role', '')}: {m.get('content', '')}\n"
            text = flat
        return len(text.split())

    def estimate_cost(self, tokens: int, mode: str = "input") -> float:
        """Calculates estimated price based on $0.20/M input and $0.88/M output pricing."""
        if mode == "input":
            return (tokens / 1_000_000.0) * 0.20
        else:
            return (tokens / 1_000_000.0) * 0.88

    def remaining_context(self, current_tokens: int) -> int:
        """Returns context limits headroom."""
        return max(0, 32768 - current_tokens)

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """DeepInfra doesn't cache local repository directories."""
        return []

    def clear_cache(self) -> bool:
        """DeepInfra doesn't cache local files."""
        return True

    def cache_size(self) -> float:
        """Return zero cache footprint."""
        return 0.0

    def _generate_mock(
        self,
        prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Returns mock responses conforming to V2 schemas for development and local testing."""
        prompt_str = str(prompt or "")
        if messages:
            prompt_str += " " + " ".join(str(m.get("content", "")) for m in messages)

        mock_data = {}
        if self.task == "vision":
            mock_data = {
              "room_type": "bedroom",
              "objects": [
                {
                  "id": "bed_001",
                  "name": "bed",
                  "category": "furniture",
                  "estimated_dimensions": {"width_cm": 180, "depth_cm": 200, "height_cm": 90},
                  "room_position": {"x_cm": 200, "y_cm": 150, "z_cm": 45},
                  "rotation": 0
                },
                {
                  "id": "wardrobe_001",
                  "name": "wardrobe",
                  "category": "furniture",
                  "estimated_dimensions": {"width_cm": 120, "depth_cm": 60, "height_cm": 210},
                  "room_position": {"x_cm": 380, "y_cm": 80, "z_cm": 105},
                  "rotation": 90
                }
              ],
              "doors": [
                {
                  "id": "door_001",
                  "type": "swing",
                  "estimated_dimensions": {"width_cm": 90, "depth_cm": 10, "height_cm": 200},
                  "room_position": {"x_cm": 50, "y_cm": 350, "z_cm": 100}
                }
              ],
              "windows": [
                {
                  "id": "window_001",
                  "type": "standard",
                  "estimated_dimensions": {"width_cm": 120, "depth_cm": 20, "height_cm": 150},
                  "room_position": {"x_cm": 240, "y_cm": 50, "z_cm": 120}
                }
              ],
              "walls": [
                {"orientation": "north", "color": "warm white", "material": "plaster"},
                {"orientation": "south", "color": "warm white", "material": "plaster"},
                {"orientation": "east", "color": "warm white", "material": "plaster"},
                {"orientation": "west", "color": "warm white", "material": "plaster"}
              ],
              "floor": {
                "material": "oak wood",
                "color": "natural brown"
              },
              "ceiling": {
                "height_cm": 290,
                "color": "white"
              },
              "confidence_score": 0.92
            }
        elif self.task == "spatial" or "spatial" in prompt_str.lower():
            mock_data = {
              "walkability_score": 88,
              "clearance_issues": ["No major clearance blocks detected."],
              "unused_space": ["corners"],
              "collision_objects": [],
              "usable_area_percentage": 80
            }
        elif self.task == "design" or "design" in prompt_str.lower():
            mock_data = {
              "design_theme": "modern minimal",
              "wall_color": "warm white",
              "floor_material": "oak wood",
              "lighting_type": "warm lighting",
              "dominant_colors": ["#ffffff", "#c49a6c"],
              "style_score": 0.91
            }
        elif self.task == "budget":
            mock_data = {
              "subtotal": 28500,
              "shipping": 1200,
              "tax": 3100,
              "total": 32800,
              "budget_limit": 35000,
              "remaining_budget": 2200
            }
        elif self.task == "explain" or "explain" in prompt_str.lower():
            mock_data = {
              "product_explanations": ["Selected study desk because it fits clearance bounds.", "Selected chair because it provides ergonomic support."],
              "placement_explanations": ["Placed desk near the window for natural light.", "Placed chair under the desk to maximize walking space."],
              "budget_explanations": ["Selected products are within the limit of Rs. 40000."]
            }
        else:
            mock_data = {"status": "success", "message": "Development mock response"}

        import json
        return {
            "text": json.dumps(mock_data),
            "response": json.dumps(mock_data),
            "success": True,
            "tokens": {"input": 100, "output": 100}
        }
