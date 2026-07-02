"""Budget Agent Module for VisionSpace AI.

Responsible for calculating total cost, budget limits, remaining reserves,
and providing budget utilization metrics alongside dynamic alternatives.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import config
from models.text_model import TextModel
from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.BudgetAgent")


class BudgetAgent:
    """Production-grade BudgetAgent responsible for spending optimizations matching V2 schemas."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the BudgetAgent."""
        self.provider_name = provider
        self.text_model = TextModel(provider=provider)
        logger.info("BudgetAgent initialized (Provider: %s)", provider)

    def analyze_budget(
        self,
        commerce_output: Dict[str, Any],
        budget_limit: float | int = 50000,
        session_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the Phase 5 budget analysis engine."""
        logger.info("Running Phase 5 Budget Agent analysis...")

        products = commerce_output.get("recommended_products") or []
        total_requested_cost = sum(float(p.get("price", 0)) for p in products)

        prompt_system = (
            "You are the Budget Agent of VisionSpace AI.\n"
            "Analyze item costs, calculate utilization metrics, and suggest alternatives.\n"
            "Strict JSON ONLY. No conversational intro/outro text. No explanations. No thinking tags."
        )

        prompt_user = (
            f"Cost Analysis:\n"
            f"- Recommended Products: {products}\n"
            f"- Budget Limit: {budget_limit}\n"
            f"- Current Total Cost: {total_requested_cost}\n\n"
            "Task:\n"
            "Format the total budget profile and identify alternative products (lower cost substitutions) "
            "for the recommended list if the total cost exceeds the limit.\n"
            "Return conforming JSON format:\n"
            "{\n"
            '  "total_cost": number,\n'
            '  "budget_limit": number,\n'
            '  "remaining_budget": number,\n'
            '  "budget_utilization_percentage": number (0-100),\n'
            '  "alternative_products": [\n'
            "    {\n"
            '      "original_name": "string",\n'
            '      "alternative_name": "string",\n'
            '      "price": number,\n'
            '      "reason": "string why recommended"\n'
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
                enable_thinking=False  # Disabled for budget extraction tasks
            )
            raw_text = llm_res.get("text") if isinstance(llm_res, dict) else str(llm_res)
            llm_data = self._validate_and_repair_json(raw_text)
        except Exception as e:
            logger.error("Budget LLM evaluation failed: %s", str(e))
            llm_data = {}

        # Fallback values if LLM fails
        fused = self._fuse_budget_payload(llm_data, total_requested_cost, budget_limit, products)

        # Save output if path is given
        if session_path is not None:
            try:
                import pathlib
                p = pathlib.Path(session_path)
                if p.is_dir():
                    save_json(fused, p / "budget_analysis.json")
            except Exception as e:
                logger.error("Failed to save budget analysis JSON: %s", str(e))

        return fused

    def _fuse_budget_payload(
        self, llm_data: Dict[str, Any], total_requested_cost: float, budget_limit: float, products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        fused = {}
        
        t_cost = float(llm_data.get("total_cost") if llm_data.get("total_cost") is not None else total_requested_cost)
        b_limit = float(llm_data.get("budget_limit") if llm_data.get("budget_limit") is not None else budget_limit)
        
        fused["total_cost"] = t_cost
        fused["budget_limit"] = b_limit
        fused["remaining_budget"] = b_limit - t_cost
        fused["budget_utilization_percentage"] = int(round((t_cost / b_limit) * 100)) if b_limit > 0 else 0
        
        alternatives = llm_data.get("alternative_products")
        if not alternatives or not isinstance(alternatives, list):
            alternatives = []
            # Propose a standard fallback alternative for the first expensive item if total cost is tight
            if products:
                first_item = products[0]
                alternatives.append({
                    "original_name": first_item.get("name", "Item"),
                    "alternative_name": f"Budget Alternative {first_item.get('category', 'Furniture')}",
                    "price": int(float(first_item.get("price", 0)) * 0.75),
                    "reason": "Lower-cost material alternative to improve budget headroom."
                })
        fused["alternative_products"] = alternatives
        
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
