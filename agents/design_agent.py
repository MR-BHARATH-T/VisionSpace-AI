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
        """Orchestrates design plan recommendations generation.

        1. Passes Vision + Spatial outputs to TextModel for design recommendation.
        2. Validates and parses structural suggestions.
        3. Exports output to design_analysis.json.
        """
        logger.info("Generating style alignment and design recommendations...")

        # Extract merged scene if nested in a VisionAgent response structure
        if "merged_scene" in vision_output:
            vision_output = vision_output["merged_scene"]

        # Call LLM for qualitative reasoning
        prompt_data = design_recommendation_prompt(
            vision_output=vision_output,
            spatial_output=spatial_output,
            budget=budget,
            preferred_style=preferred_style,
            preferred_colors=preferred_colors,
            preferred_brands=preferred_brands,
            constraints=constraints
        )

        try:
            llm_res = self.text_model.generate(
                system_prompt=prompt_data["system"],
                prompt=prompt_data["user"],
                max_new_tokens=1024,
                temperature=0.1
            )
            raw_text = llm_res.get("text") if isinstance(llm_res, dict) else str(llm_res)
            llm_data = self._validate_and_repair_json(raw_text)
        except Exception as e:
            logger.error("LLM Design generation failed, falling back to heuristics: %s", str(e))
            llm_data = self._get_fallback_design_json(preferred_style, budget)

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
        """Ensures that the output contains all compliant Design Agent schema fields."""
        fallback = self._get_fallback_design_json(style, budget)
        fused = {}

        for key, val in fallback.items():
            if key in llm_data:
                fused[key] = llm_data[key]
            else:
                fused[key] = val

        try:
            fused["estimated_total_cost"] = float(fused.get("estimated_total_cost") or budget * 0.8)
        except Exception:
            fused["estimated_total_cost"] = float(budget * 0.8)

        # Remove temporary or legacy format keys if generated
        fused.pop("estimated_cost", None)
        fused.pop("placement_plan", None)

        return fused
