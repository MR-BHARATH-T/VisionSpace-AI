"""Explanation Agent Module for VisionSpace AI.

Responsible for generating human-readable layout, budget, and product
recommendation justifications conformed to word count constraints.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import config
from models.text_model import TextModel
from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.ExplanationAgent")


class ExplanationAgent:
    """Production-grade ExplanationAgent responsible for qualitative rationale generation matching V2 schemas."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the ExplanationAgent."""
        self.provider_name = provider
        self.text_model = TextModel(provider=provider)
        logger.info("ExplanationAgent initialized (Provider: %s)", provider)

    def generate_explanations(
        self,
        vision_output: Dict[str, Any],
        spatial_output: Dict[str, Any],
        design_output: Dict[str, Any],
        commerce_output: Dict[str, Any],
        budget_output: Dict[str, Any],
        session_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the Phase 6 explanation engine."""
        logger.info("Running Phase 6 Explanation Agent justifications...")

        products = commerce_output.get("recommended_products") or []
        budget_limit = budget_output.get("budget_limit", 50000)
        total_cost = budget_output.get("total_cost", 0)

        prompt_system = (
            "You are the Explanation Agent of VisionSpace AI. Generate structured explanations.\n"
            "Analyze layouts, selections, and budget parameters.\n"
            "Word limit: Under 50 words per product explanation.\n"
            "Strict JSON ONLY. No conversational intro/outro text. No explanations. No thinking tags in output JSON."
        )

        prompt_user = (
            f"Context details:\n"
            f"- Recommended Products: {products}\n"
            f"- Usable Space Percentage: {spatial_output.get('usable_area_percentage', 80)}\n"
            f"- Lighting Score: {design_output.get('lighting_score', 80)}\n"
            f"- Budget Utilization: {budget_output.get('budget_utilization_percentage', 90)}% "
            f"({total_cost} of {budget_limit})\n\n"
            "Task:\n"
            "Generate short explanations justifying the selection of each product, placement choice, "
            "and budget optimization. Do not exceed 50 words per product explanation!\n"
            "Return conforming JSON format:\n"
            "{\n"
            '  "product_explanations": ["list of strings justifying selected/rejected items"],\n'
            '  "placement_explanations": ["list of strings justifying why objects were located in specific regions"],\n'
            '  "budget_explanations": ["list of strings detailing budget alignment/optimizations"]\n'
            "}"
        )

        try:
            llm_res = self.text_model.generate(
                system_prompt=prompt_system,
                prompt=prompt_user,
                max_new_tokens=1024,
                temperature=0.1,
                enable_thinking=True  # Enabled for qualitative reasoning tasks
            )
            raw_text = llm_res.get("text") if isinstance(llm_res, dict) else str(llm_res)
            llm_data = self._validate_and_repair_json(raw_text)
        except Exception as e:
            logger.error("Explanation LLM evaluation failed: %s", str(e))
            llm_data = {}

        # Fallback values if LLM fails
        fused = self._fuse_explanation_payload(llm_data, products)

        # Save output if path is given
        if session_path is not None:
            try:
                import pathlib
                p = pathlib.Path(session_path)
                if p.is_dir():
                    save_json(fused, p / "explain_analysis.json")
            except Exception as e:
                logger.error("Failed to save explain analysis JSON: %s", str(e))

        return fused

    def _fuse_explanation_payload(
        self, llm_data: Dict[str, Any], products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        fused = {}
        
        pe = llm_data.get("product_explanations")
        if not pe or not isinstance(pe, list):
            pe = []
            for p in products:
                pe.append(f"The {p.get('name', 'item')} is selected because it fits the modern style and fulfills the user functional requirement.")
        fused["product_explanations"] = pe

        ple = llm_data.get("placement_explanations")
        if not ple or not isinstance(ple, list):
            ple = ["The desk is placed next to the window to maximize natural light usage while leaving the central path clear."]
        fused["placement_explanations"] = ple

        be = llm_data.get("budget_explanations")
        if not be or not isinstance(be, list):
            be = ["All items stay within boundaries, achieving optimal utility and style within the budget constraint."]
        fused["budget_explanations"] = be

        return fused

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
