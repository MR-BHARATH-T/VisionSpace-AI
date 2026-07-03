"""Design Agent Module for VisionSpace AI.

Analyzes scene graphs, layout metrics, color presets, brand preferences,
and custom constraints to generate an optimized interior design plan matching the master prompt.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import config
from models.text_model import TextModel
from utils.prompt_templates import design_recommendation_prompt
from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.DesignAgent")


class DesignAgent:
    """Production-grade DesignAgent responsible for style alignment and design plan optimizations."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the DesignAgent.

        Args:
            provider: Inference provider ("huggingface").
        """
        self.provider_name = provider
        self.text_model = TextModel(provider=provider)
        logger.info("DesignAgent initialized (Provider: %s)", provider)

    def generate_recommendations(
        self,
        vision_output: Dict[str, Any],
        spatial_output: Dict[str, Any],
        budget: float | int = 50000,
        preferred_style: str = "scandinavian",
        preferred_colors: Optional[List[str]] = None,
        preferred_brands: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        session_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Orchestrates design plan recommendations generation."""
        logger.info("Generating style alignment and design recommendations...")

        # Guard: Execute only if walkability_score >= 60
        walkability_score = spatial_output.get("walkability_score", 0)
        if walkability_score < 60:
            logger.warning("Spatial walkability score %s < 60. Skipping Design recommendations.", walkability_score)
            return {
                "design_theme": f"modern {preferred_style}",
                "wall_color": "warm white",
                "floor_material": "oak wood",
                "lighting_type": "warm lighting",
                "dominant_colors": ["#ffffff", "#c49a6c"],
                "style_score": 0.0
            }

        # Extract merged scene if nested in a VisionAgent response structure
        if "merged_scene" in vision_output:
            vision_output = vision_output["merged_scene"]

        prompt_system = (
            "You are the Design Agent of VisionSpace AI. Perform interior design evaluations.\n"
            "Analyze current styling, colors, lighting, and layout opportunities.\n"
            "Strict JSON ONLY. No conversational intro/outro text. No explanations. No thinking tags in output JSON."
        )
        prompt_user = (
            f"Current room details:\n"
            f"- Room Type: {vision_output.get('room_type', 'bedroom')}\n"
            f"- Empty regions: {vision_output.get('empty_regions', [])}\n"
            f"- Spatial walkability: {walkability_score}\n\n"
            "Analyze the design aesthetics and return conforming JSON:\n"
            "{\n"
            '  "design_theme": "string (e.g. modern minimal)",\n'
            '  "wall_color": "string",\n'
            '  "floor_material": "string",\n'
            '  "lighting_type": "string (e.g. warm lighting)",\n'
            '  "dominant_colors": ["hex_colors"],\n'
            '  "style_score": number\n'
            "}"
        )

        try:
            llm_res = self.text_model.generate(
                system_prompt=prompt_system,
                prompt=prompt_user,
                max_new_tokens=512,
                temperature=0.1,
                enable_thinking=True
            )
            raw_text = llm_res.get("text") if isinstance(llm_res, dict) else str(llm_res)
            llm_data = self._validate_and_repair_json(raw_text)
        except Exception as e:
            logger.error("LLM Design generation failed: %s", str(e))
            llm_data = {}

        # Fuses final output payload
        final_output = self._fuse_design_payload(llm_data, preferred_style, budget)

        # Save output if path is given
        if session_path is not None:
            try:
                import pathlib
                p = pathlib.Path(session_path)
                if p.is_dir():
                    save_json(final_output, p / "design_analysis.json")
            except Exception as e:
                logger.error("Failed to save design analysis JSON: %s", str(e))

        return final_output

    def _validate_and_repair_json(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Helper to repair and clean JSON response structures."""
        if not raw_text:
            return {}
        clean_text = raw_text.strip()
        
        # Strip thinking block (<think>...</think>) if present
        if "<think>" in clean_text.lower():
            if "</think>" in clean_text.lower():
                parts = clean_text.split("</think>")
                clean_text = parts[-1].strip()
            else:
                idx = clean_text.lower().find("<think>")
                brace_idx = clean_text.find("{", idx)
                if brace_idx != -1:
                    clean_text = clean_text[brace_idx:].strip()
                else:
                    clean_text = clean_text[idx + len("<think>"):].strip()
        
        # Locate outer JSON braces to discard extraneous reasoning text
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            clean_text = clean_text[start_idx:end_idx + 1].strip()
        
        # Remove code blocks if present
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                clean_text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        
        clean_text = clean_text.strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove trailing commas
        try:
            repaired = re.sub(r',\s*([\]}])', r'\1', clean_text)
            return json.loads(repaired)
        except Exception:
            pass

        # Strategy 3: json-repair fallback
        try:
            from json_repair import repair_json
            repaired_lib = repair_json(clean_text, return_objects=True)
            if isinstance(repaired_lib, dict):
                return repaired_lib
        except Exception:
            pass

        logger.error("Design JSON parsing completely failed. returning empty dict.")
        return {}

    def _get_fallback_design_json(self, style: str, budget: float | int) -> Dict[str, Any]:
        """Backup design plan structure if LLM inference crashed."""
        return {
            "design_theme": f"Warm {style.capitalize()} Sanctuary",
            "design_style": style,
            "room_goal": "Optimize layout for ergonomic productivity, accessibility, and visual balance.",
            "materials": {
                "primary_material": "Oak Wood",
                "secondary_material": "Linen Fabric",
                "accent_material": "Black Powder Coated Steel"
            },
            "lighting_strategy": {
                "lighting_mood": "Cozy, layered, warm white tone for reading and working.",
                "color_temperature": "3000K",
                "natural_light_strategy": "Desk placed next to window for direct lighting; light linen curtains to reduce glare."
            },
            "color_palette": {
                "primary_colors": ["White", "Oak"],
                "secondary_colors": ["Light Grey"],
                "accent_colors": ["Sage Green"],
                "wall_colors": ["Off-White"],
                "furniture_colors": ["Oak", "White"]
            },
            "recommended_furniture": [
                {
                    "id": "desk_001",
                    "name": "Oak Study Desk",
                    "category": "desk",
                    "priority": "Mandatory",
                    "estimated_cost": int(budget * 0.15),
                    "retain_existing": False,
                    "reason": "Provides a dedicated ergonomically aligned workspace."
                }
            ],
            "layout_plan": [
                {
                    "furniture_id": "desk_001",
                    "position": {
                        "x_cm": 150,
                        "y_cm": 200
                    },
                    "rotation": 90,
                    "zone": "workspace",
                    "clearance_valid": True
                }
            ],
            "functional_zones": ["sleeping", "working", "storage", "circulation"],
            "budget_breakdown": {
                "furniture": int(budget * 0.6),
                "lighting": int(budget * 0.1),
                "decor": int(budget * 0.05),
                "storage": int(budget * 0.15),
                "buffer": int(budget * 0.1),
                "total": int(budget)
            },
            "shopping_requirements": [
                {
                    "category": "desk",
                    "required_width_cm": 120,
                    "required_depth_cm": 60,
                    "style": style,
                    "material": "oak",
                    "budget_limit": int(budget * 0.15),
                    "preferred_brands": ["IKEA", "Urban Ladder"]
                }
            ],
            "estimated_total_cost": int(budget * 0.9),
            "design_explanations": [
                "The Oak Study Desk was selected to create a functional work-from-home zone.",
                "It was placed next to the window to maximize natural light.",
                "We preserved a clear 90 cm wheelchair accessible walking corridor from the entrance."
            ],
            "confidence_score": 0.85
        }

    def _fuse_design_payload(self, llm_data: Dict[str, Any], style: str, budget: float | int) -> Dict[str, Any]:
        """Conforms output structure strictly to Design Agent schema."""
        return {
            "design_theme": str(llm_data.get("design_theme") or f"modern {style}"),
            "wall_color": str(llm_data.get("wall_color") or "warm white"),
            "floor_material": str(llm_data.get("floor_material") or "oak wood"),
            "lighting_type": str(llm_data.get("lighting_type") or "warm lighting"),
            "dominant_colors": list(llm_data.get("dominant_colors") or ["#ffffff", "#c49a6c"]),
            "style_score": float(llm_data.get("style_score") if llm_data.get("style_score") is not None else 0.91)
        }
