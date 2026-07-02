"""Commerce/Product Agent Module for VisionSpace AI.

Coordinates user preferences with room inventory, spatial clearances, and design styles
to output filtered product recommendations (requested products + complementary budget-aligned items).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import config
from models.text_model import TextModel
from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.CommerceAgent")


class CommerceAgent:
    """Production-grade CommerceAgent responsible for item curation matching V2 schemas."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the CommerceAgent."""
        self.provider_name = provider
        self.text_model = TextModel(provider=provider)
        logger.info("CommerceAgent initialized (Provider: %s)", provider)

    def recommend_products(
        self,
        vision_output: Dict[str, Any],
        spatial_output: Dict[str, Any],
        design_output: Dict[str, Any],
        user_preferences: Dict[str, Any],
        session_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the Phase 4 recommendation engine."""
        logger.info("Running Phase 4 Product Agent recommendations...")

        # Guard: Ensure preferences structure exists
        req_products = user_preferences.get("required_products") or []
        preferred_brands = user_preferences.get("preferred_brands") or []
        excluded_brands = user_preferences.get("excluded_brands") or []
        budget_range = user_preferences.get("budget_range") or {"minimum": 0, "maximum": 50000}
        max_budget = budget_range.get("maximum") or 50000
        pref_style = user_preferences.get("preferred_style") or "modern"

        prompt_system = (
            "You are the Product Agent of VisionSpace AI.\n"
            "Recommend products matching user requirements, preferred brands, and room limits.\n"
            "Strict JSON ONLY. No conversational intro/outro text. No explanations. No thinking tags."
        )

        prompt_user = (
            f"Requirements:\n"
            f"- User Requested Products: {req_products}\n"
            f"- Preferred Brands: {preferred_brands}\n"
            f"- Excluded Brands: {excluded_brands}\n"
            f"- Total Budget Limit: {max_budget}\n"
            f"- Styling Preference: {pref_style}\n\n"
            f"Context:\n"
            f"- Room Type: {vision_output.get('room_type', 'bedroom')}\n"
            f"- Walkability Score: {spatial_output.get('walkability_score', 80)}\n"
            f"- Style Style: {design_output.get('current_style', pref_style)}\n\n"
            "Task:\n"
            "Recommend the user-requested products, plus some complementary useful furniture/decor items "
            "that fit the style and stay strictly within the budget.\n"
            "Exclude any brands on the excluded list.\n"
            "Return conforming JSON format:\n"
            "{\n"
            '  "recommended_products": [\n'
            "    {\n"
            '      "name": "string product name",\n'
            '      "category": "string category",\n'
            '      "brand": "string brand name",\n'
            '      "price": number,\n'
            '      "reason": "string justification why selected",\n'
            '      "fit_score": number (0.0 - 1.0),\n'
            '      "space_score": number (0.0 - 1.0),\n'
            '      "budget_score": number (0.0 - 1.0)\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            llm_res = self.text_model.generate(
                system_prompt=prompt_system,
                prompt=prompt_user,
                max_new_tokens=1024,
                temperature=0.1,
                enable_thinking=False  # Disabled for detection/extraction tasks
            )
            raw_text = llm_res.get("text") if isinstance(llm_res, dict) else str(llm_res)
            llm_data = self._validate_and_repair_json(raw_text)
        except Exception as e:
            logger.error("Commerce LLM recommendation failed: %s", str(e))
            llm_data = {}

        # Fallback recommendations if LLM fails
        fused = self._fuse_commerce_payload(llm_data, req_products, preferred_brands, max_budget)

        # Save output if path is given
        if session_path is not None:
            try:
                import pathlib
                p = pathlib.Path(session_path)
                if p.is_dir():
                    save_json(fused, p / "commerce_analysis.json")
            except Exception as e:
                logger.error("Failed to save commerce analysis JSON: %s", str(e))

        return fused

    def _fuse_commerce_payload(
        self, llm_data: Dict[str, Any], req_products: List[str], preferred_brands: List[str], max_budget: float
    ) -> Dict[str, Any]:
        products = llm_data.get("recommended_products")
        if not products or not isinstance(products, list):
            # Create a simple default fallback matching requirements
            products = []
            default_brand = preferred_brands[0] if preferred_brands else "IKEA"
            
            # 1. Add requested items
            for idx, req in enumerate(req_products):
                products.append({
                    "name": f"{default_brand} {req.capitalize()}",
                    "category": req,
                    "brand": default_brand,
                    "price": int(max_budget * 0.2),
                    "reason": f"Matches user requested {req}.",
                    "fit_score": 0.95,
                    "space_score": 0.90,
                    "budget_score": 0.90
                })
            
            # 2. Add a useful complementary product if space permits
            products.append({
                "name": f"{default_brand} LED Desk Lamp",
                "category": "lighting",
                "brand": default_brand,
                "price": int(max_budget * 0.05),
                "reason": "Complementary task lighting to boost workspace utility.",
                "fit_score": 0.90,
                "space_score": 0.95,
                "budget_score": 0.95
            })

        return {"recommended_products": products}

    def _validate_and_repair_json(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Helper to repair and clean JSON response structures."""
        if not raw_text:
            return {}
        clean_text = raw_text.strip()
        
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

        try:
            repaired = re.sub(r',\s*([\]}])', r'\1', clean_text)
            return json.loads(repaired)
        except Exception:
            pass

        try:
            from json_repair import repair_json
            repaired_lib = repair_json(clean_text, return_objects=True)
            if isinstance(repaired_lib, dict):
                return repaired_lib
        except Exception:
            pass

        return {}
