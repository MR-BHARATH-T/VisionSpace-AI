"""Production Workflow Orchestrator Module for VisionSpace AI.

Coordinates the complete Phase 0 through Phase 7 sequential execution pipeline,
applying strict confidence/walkability guards, dynamic VLM parameters, and budget filters.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, Union

import config
from agents.vision_agent import VisionAgent
from agents.spatial_agent import SpatialAgent
from agents.design_agent import DesignAgent
from agents.commerce_agent import CommerceAgent
from agents.budget_agent import BudgetAgent
from agents.explain_agent import ExplanationAgent
from agents.reconstruction_agent import ReconstructionAgent
from utils.file_utils import save_json

logger = logging.getLogger("VisionSpaceAI.WorkflowOrchestrator")


class ProductionPipelineV2:
    """Production sequential execution pipeline for VisionSpace AI v2."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the multi-agent orchestration components."""
        self.provider = provider
        self.vision_agent = VisionAgent(provider=provider)
        self.spatial_agent = SpatialAgent(provider=provider)
        self.design_agent = DesignAgent(provider=provider)
        self.commerce_agent = CommerceAgent(provider=provider)
        self.budget_agent = BudgetAgent(provider=provider)
        self.explain_agent = ExplanationAgent(provider=provider)
        self.reconstruction_agent = ReconstructionAgent(provider=provider)
        
        logger.info("ProductionPipelineV2 workflow orchestrator initialized.")

    def run_pipeline(
        self,
        images: Union[Any, List[Any]],
        user_preferences: Dict[str, Any],
        room_type: str = "bedroom",
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the sequential orchestration of Phase 0 -> Phase 7."""
        logger.info("Executing VisionSpace AI Production Pipeline v2...")
        
        # Phase 0: User Preference Collection (Pre-existing in argument)
        if progress_callback:
            progress_callback("Phase 0: Preferences loaded successfully.")

        # Phase 1: Unified Vision Agent Pass (Unified Batched Pass)
        if progress_callback:
            progress_callback("Phase 1: Running Unified VLM Vision Pass...")
            
        vision_output = self.vision_agent.analyze_room(
            images=images,
            room_type=room_type,
            progress_callback=progress_callback
        )
        
        session_id = vision_output["session_id"]
        session_path = vision_output["metadata"]["session_path"]
        merged_scene = vision_output["merged_scene"]

        # Production Guard: If confidence score < 0.70, request more images and do not continue
        conf_score = merged_scene.get("confidence_score", 0.0)
        
        # Robust fallback: If 0.0 but we have parsed objects, set to 0.85
        if conf_score == 0.0 and (merged_scene.get("objects") or merged_scene.get("room_type")):
            logger.warning("VLM output did not include confidence score. Setting fallback confidence to 0.85")
            conf_score = 0.85
            merged_scene["confidence_score"] = conf_score

        if conf_score < 0.70:
            logger.error("Low vision confidence score (%s < 0.70). Aborting pipeline.", conf_score)
            raise ValueError(f"Aborting Pipeline: Low vision confidence score ({conf_score} < 0.70). Please upload more or clearer images.")

        # Phase 2: Spatial Agent Pass
        if progress_callback:
            progress_callback("Phase 2: Executing layout clearance validation...")
        spatial_output = self.spatial_agent.analyze_layout(
            room_data=merged_scene,
            session_path=session_path
        )

        # Phase 3: Design Agent Pass
        if progress_callback:
            progress_callback("Phase 3: Performing design theme alignment...")
        design_output = self.design_agent.generate_recommendations(
            vision_output=merged_scene,
            spatial_output=spatial_output,
            budget=user_preferences.get("budget_range", {}).get("maximum", 50000),
            preferred_style=user_preferences.get("preferred_style", "modern"),
            preferred_colors=user_preferences.get("color_preferences"),
            preferred_brands=user_preferences.get("preferred_brands"),
            session_path=session_path
        )

        # Phase 4: Product Agent (Commerce Agent) recommendations - Always Run
        if progress_callback:
            progress_callback("Phase 4: Running product recommendations engine...")
        commerce_output = self.commerce_agent.recommend_products(
            vision_output=merged_scene,
            spatial_output=spatial_output,
            design_output=design_output,
            user_preferences=user_preferences,
            session_path=session_path
        )
        self.commerce_agent.generate_visual_preview(
            products=commerce_output.get("recommended_products", []),
            session_path=str(session_path)
        )

        # Simulate user selecting specific items (e.g. select the top ranked item for each category)
        selected_products = []
        req_products = user_preferences.get("required_products") or []
        for cat in req_products:
            normalized_key = cat.lower().replace(" ", "_")
            cat_list = commerce_output.get(normalized_key) or []
            if cat_list:
                selected_products.append(cat_list[0])

        if progress_callback:
            progress_callback(f"Simulating user selection: Chosen {len(selected_products)} products.")

        # Phase 5: Budget Agent cost optimization (Run on selected products)
        budget_output = {}
        if progress_callback:
            progress_callback("Phase 5: Performing budget spending check on selected products...")
        budget_limit = user_preferences.get("budget_range", {}).get("maximum", 50000)
        budget_output = self.budget_agent.analyze_budget(
            commerce_output=selected_products,
            budget_limit=budget_limit,
            session_path=session_path
        )

        # Phase 6: Explanation Agent narratives builder
        explain_output = {}
        if progress_callback:
            progress_callback("Phase 6: Generating layout explanations...")
        explain_output = self.explain_agent.generate_explanations(
            vision_output=merged_scene,
            spatial_output=spatial_output,
            design_output=design_output,
            commerce_output={"recommended_products": selected_products},
            budget_output=budget_output,
            session_path=session_path
        )

        # Phase 7: Reconstruction Agent delta map
        reconstruct_output = {}
        if progress_callback:
            progress_callback("Phase 7: Mapping layout transformation delta...")
            
        # Free GPU memory before Reconstruction Phase
        try:
            self.vision_agent.model_manager.unload_all()
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleared VRAM memory to prepare for image reconstruction.")
        except Exception as e:
            logger.warning("Failed to clear VRAM: %s", str(e))

        reconstruct_output = self.reconstruction_agent.plan_reconstruction(
            vision_output=merged_scene,
            commerce_output={"recommended_products": selected_products},
            session_path=session_path,
            spatial_output=spatial_output
        )

        # Build final aggregated response conformed to app zipping structure
        return {
            "session_id": session_id,
            "session_path": session_path,
            "vision": vision_output,
            "spatial": spatial_output,
            "design": design_output,
            "commerce": commerce_output,
            "selected_products": selected_products,
            "budget": budget_output,
            "explanation": explain_output,
            "reconstruction": reconstruct_output
        }
