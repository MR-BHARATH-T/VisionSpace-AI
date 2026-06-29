# utils/prompt_templates.py
"""
VisionSpace AI — Production Prompt Templates
============================================
Centralised, structured prompt builders for every VisionSpace agent.

Design principles:
  - Every template is a *function*, not a raw string — parameters are
    explicit and type-checked.
  - Templates produce structured prompts that request JSON responses so
    agents can be parsed reliably.
  - Each builder has a ``system`` + ``user`` key to map cleanly to chat APIs.
  - Section markers (### CONTEXT, ### TASK, ### OUTPUT FORMAT) are consistent
    across all templates for easy prompt engineering iteration.

Usage::

    from utils.prompt_templates import vision_analysis_prompt
    messages = vision_analysis_prompt(room_type="bedroom")
    # messages → {"system": "...", "user": "..."}
"""

from __future__ import annotations

import json
from typing import Any

from utils.constants import DESIGN_STYLES, ROOM_TYPES


# ---------------------------------------------------------------------------
# Shared Boilerplate
# ---------------------------------------------------------------------------

_JSON_REMINDER = (
    "Respond ONLY with a valid JSON object matching the schema above. "
    "Do not include markdown code fences, explanations, or any text outside the JSON."
)

_EXPERT_IDENTITY = (
    "You are VisionSpace AI, an expert interior design assistant powered by "
    "computer vision. You analyse rooms with professional precision and provide "
    "actionable, commercially-aware recommendations."
)


def _section(title: str, body: str) -> str:
    return f"### {title}\n{body.strip()}\n"


# ---------------------------------------------------------------------------
# 1. Vision Agent — Scene Analysis
# ---------------------------------------------------------------------------

def vision_analysis_prompt(
    room_type: str = "living_room",
    confidence_threshold: float = 0.65,
    max_objects: int = 30,
) -> dict[str, str]:
    """Prompt for the Vision Agent to generate a comprehensive scene understanding analysis.

    The model should return a highly detailed structured JSON matching the 25-feature
    interior design schema, including room attributes, structural details, object details,
    spatial relationships, floor/wall/ceiling properties, lighting, free space, and safety hazards.

    Args:
        room_type:            Room category hint for context.
        confidence_threshold: Minimum detection confidence to include for objects.
        max_objects:          Upper limit on reported objects.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    schema = {
        "room_understanding": {
            "room_type": "string (e.g., living_room, bedroom)",
            "room_shape": "string (e.g., rectangular, L-shaped)",
            "room_size": "string (small|medium|large)",
            "room_function": "string",
            "estimated_purpose": "string",
            "occupancy": "string",
            "architectural_style": "string",
            "ceiling_height_cm": "integer|null",
            "ceiling_type": "string|null",
            "ceiling_color": "string|null",
            "wall_finish": "string|null",
            "floor_finish": "string|null",
            "interior_theme": "string|null"
        },
        "doors": [
            {
                "id": "string",
                "type": "string (swing|sliding|folding|other)",
                "opens": "string (inside|outside|unknown)",
                "estimated_width_cm": "number|null",
                "estimated_height_cm": "number|null",
                "dimension_confidence": "number|null (0.0-1.0)",
                "measurement_source": "string (vision_estimation)",
                "blocked": "boolean|null",
                "room_position": {
                    "x_cm": "number|null",
                    "y_cm": "number|null",
                    "rotation": "integer|null",
                    "depth_confidence": "number|null (0.0-1.0)",
                    "position_source": "string (vision_estimation|relative_scale_estimation)"
                },
                "rotation_deg": "integer|null",
                "bbox": {
                    "x": "number",
                    "y": "number",
                    "width": "number",
                    "height": "number"
                }
            }
        ],
        "windows": [
            {
                "id": "string",
                "type": "string",
                "estimated_width_cm": "number|null",
                "estimated_height_cm": "number|null",
                "dimension_confidence": "number|null (0.0-1.0)",
                "measurement_source": "string (vision_estimation)",
                "natural_light_contribution": "string (high|medium|low)",
                "curtains": "boolean|null",
                "blinds": "boolean|null",
                "glass_type": "string|null",
                "room_position": {
                    "x_cm": "number|null",
                    "y_cm": "number|null",
                    "rotation": "integer|null",
                    "depth_confidence": "number|null (0.0-1.0)",
                    "position_source": "string (vision_estimation|relative_scale_estimation)"
                },
                "rotation_deg": "integer|null",
                "bbox": {
                    "x": "number",
                    "y": "number",
                    "width": "number",
                    "height": "number"
                }
            }
        ],
        "image_metadata": [
            {
                "image_index": "integer",
                "camera_view": "string",
                "estimated_camera_angle": "number|null",
                "image_width": "integer",
                "image_height": "integer",
                "orientation": "string (landscape|portrait)"
            }
        ],
        "scene_coverage_percentage": "integer|null",
        "floor_analysis": {
            "material": "string|null",
            "texture": "string|null",
            "reflection": "string|null",
            "pattern": "string|null",
            "color": "string|null",
            "condition": "string|null"
        },
        "wall_analysis": {
            "finish": "string|null",
            "color": "string|null",
            "texture": "string|null",
            "decorations": ["string"],
            "paint_condition": "string|null"
        },
        "ceiling_analysis": {
            "ceiling_type": "string|null",
            "ceiling_color": "string|null",
            "ceiling_lights": ["string"],
            "false_ceiling_present": "boolean|null",
            "pop": "boolean|null",
            "wooden": "boolean|null",
            "concrete": "boolean|null"
        },
        "objects": [
            {
                "id": "string (unique, e.g., chair_001)",
                "name": "string (e.g., Office Chair)",
                "category": "string (e.g., seating)",
                "subcategory": "string (e.g., ergonomic)",
                "confidence": "number (0.0-1.0)",
                "material": "string|null",
                "color": "string|null",
                "style": "string|null",
                "estimated_dimensions": {
                    "width_cm": "number|null",
                    "depth_cm": "number|null",
                    "height_cm": "number|null",
                    "dimension_confidence": "number|null (0.0-1.0)",
                    "measurement_source": "string (vision_estimation)"
                },
                "estimated_weight": "string (light|medium|heavy)|null",
                "movable": "boolean|null",
                "wall_aligned": "boolean|null",
                "room_position": {
                    "x_cm": "number|null",
                    "y_cm": "number|null",
                    "rotation": "integer|null",
                    "depth_confidence": "number|null (0.0-1.0)",
                    "position_source": "string (vision_estimation|relative_scale_estimation)"
                },
                "rotation_deg": "integer|null",
                "bbox": {
                    "x": "number",
                    "y": "number",
                    "width": "number",
                    "height": "number"
                },
                "segmentation_available": "boolean|null",
                "mask_available": "boolean|null",
                "visibility": "string (full|partial|occluded)|null",
                "occluded_by": "string|null (object ID)",
                "condition": "string (good|worn|damaged|unknown)|null",
                "detection_confidence": "number|null",
                "attribute_confidence": "number|null",
                "geometry_confidence": "number|null"
            }
        ],
        "spatial_relationships": [
            {
                "object1": "string (object ID)",
                "object2": "string (object ID)",
                "relation": "string (left_of|right_of|above|below|touching|overlapping|against_wall|near_window|near_door|blocking|in_front_of|behind)"
            }
        ],
        "lighting_analysis": {
            "overall": "string|null",
            "natural": "string|null",
            "artificial": "string|null",
            "brightness_score": "integer (0-100)|null",
            "shadow_level": "string (low|medium|high)|null",
            "lighting_temperature": "string (warm|neutral|cool)|null"
        },
        "free_space_analysis": {
            "percentage": "number|null",
            "usable_area_sqm": "number|null",
            "largest_rectangle": {
                "width_cm": "number|null",
                "depth_cm": "number|null"
            },
            "free_space_confidence": "number|null (0.0-1.0)"
        },
        "accessibility": {
            "walking_width_cm": "number|null",
            "wheelchair_accessible": "boolean|null",
            "trip_hazards": ["string"]
        },
        "walking_paths": [
            [
                { "x": "number", "y": "number" }
            ]
        ],
        "layout_problems": [
            {
                "severity": "string (low|medium|high)",
                "type": "string",
                "description": "string"
            }
        ],
        "empty_regions": [
            {
                "id": "string",
                "width_cm": "number",
                "depth_cm": "number",
                "suitable_for": ["string"]
            }
        ],
        "candidate_furniture_locations": [
            {
                "location": "string",
                "suitable_for": "string"
            }
        ],
        "style": "string|null",
        "dominant_color_palette": ["string"],
        "overall_aesthetic": "string|null",
        "image_quality": [
            {
                "image_index": "integer",
                "blur": "string (low|medium|high)|null",
                "noise": "string (low|medium|high)|null",
                "exposure": "string (good|underexposed|overexposed)|null",
                "usable": "boolean|null"
            }
        ],
        "safety_analysis": [
            {
                "hazard_type": "string (exposed wires|broken furniture|slippery floor|other)",
                "severity": "string (low|medium|high)",
                "description": "string"
            }
        ]
    }

    system = _expert_system_prompt()
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Room type: **{room_type}**\n"
            f"Analyze the visible environment to construct a detailed scene graph.\n"
            f"Include only objects with confidence ≥ {confidence_threshold:.2f}.\n"
            f"Report up to {max_objects} objects."
        )),
        _section("TASK", (
            "Analyze one or more images of the same room and produce a comprehensive structured "
            "understanding. Identify the room functions, style, image details, floors/walls/ceilings, "
            "all furniture/objects (unique IDs, bboxes, dimensions, visibility, occlusion, styles), "
            "spatial relationships between objects, structural features (doors, windows), "
            "lighting qualities, usability clearances, accessible paths, empty regions, and hazards."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 2. Spatial Agent — Layout Analysis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Spatial Agent — Layout Analysis
# ---------------------------------------------------------------------------

def spatial_analysis_prompt(
    room_data: dict[str, Any],
    geometric_report: dict[str, Any],
    proposed_furniture: list[dict] | None = None,
) -> dict[str, str]:
    """Prompt for the Spatial Agent to evaluate layout quality, clearances, and placements.

    Args:
        room_data:          Current room layout data (objects, doors, windows, etc.).
        geometric_report:    Factual geometric checks (collisions, clearances, A* paths, balance metrics).
        proposed_furniture:  Optional list of new furniture proposed to fit into the room.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    room_type = room_data.get("room_understanding", {}).get("room_type", "living_room")
    objects = room_data.get("objects", [])
    doors = room_data.get("doors", [])
    windows = room_data.get("windows", [])
    
    # Text summaries for VLM context
    object_summary = _summarise_objects(objects)
    door_summary = _summarise_list(doors)
    window_summary = _summarise_list(windows)
    prop_summary = _summarise_list(proposed_furniture) if proposed_furniture else "  (none)"
    
    # Geometric checks summary
    collisions = geometric_report.get("collision_report", [])
    clearances = geometric_report.get("clearance_violations", [])
    paths = geometric_report.get("walking_paths", [])
    candidates = geometric_report.get("placement_candidates", [])
    balance = geometric_report.get("balance_score_heuristic", 85)
    symmetry = geometric_report.get("symmetry_score_heuristic", 80)
    
    geom_context = (
        f"- Collision Overlaps Detected: {len(collisions)}\n"
        f"- Clearance Rule Violations: {len(clearances)}\n"
        f"- Calculated Walking Paths: {len(paths)}\n"
        f"- Placement Candidates Found: {len(candidates)}\n"
        f"- Computed Mass Balance Score: {balance:.1f} / 100\n"
        f"- Computed Symmetry Score: {symmetry:.1f} / 100\n"
        f"- Details:\n"
        f"  - Collisions: {[c['description'] for c in collisions[:5]]}\n"
        f"  - Clearances: {[c['description'] for c in clearances[:5]]}"
    )

    schema = {
        "layout_analysis": {
            "layout_score": "integer (0-100)",
            "flow_score": "integer (0-100)",
            "accessibility_score": "integer (0-100)",
            "balance_score": "integer (0-100)",
            "space_utilization_score": "integer (0-100)",
            "focal_point": "string",
            "traffic_flow": "string",
            "functional_zones": ["string"]
        },
        "collision_report": [
            {
                "object1": "string (ID)",
                "object2": "string (ID or door/window)",
                "severity": "string (high|medium)",
                "description": "string"
            }
        ],
        "clearance_violations": [
            {
                "object": "string (ID)",
                "rule": "string",
                "required_clearance_cm": "integer",
                "actual_clearance_cm": "integer",
                "description": "string"
            }
        ],
        "walking_paths": [
            {
                "name": "string (e.g. entrance_to_bed)",
                "walkable": "boolean",
                "length_cm": "integer",
                "bottlenecks": ["string"],
                "coordinates": [{"x": "integer", "y": "integer"}]
            }
        ],
        "placement_candidates": [
            {
                "id": "string",
                "center": {"x_cm": "integer", "y_cm": "integer"},
                "max_dimensions": {"width_cm": "integer", "depth_cm": "integer"},
                "fit_confidence": "number (0.0-1.0)"
            }
        ],
        "furniture_fit_validation": [
            {
                "proposed_id": "string",
                "fits": "boolean",
                "violations": ["string"],
                "preserves_accessibility": "boolean",
                "recommended_position": {"x_cm": "integer", "y_cm": "integer", "rotation": "integer"}
            }
        ],
        "recommendations": [
            {
                "action": "string (move|rotate|relocate|add|remove)",
                "object_id": "string",
                "description": "string",
                "impact": "string (high|medium|low)"
            }
        ],
        "safety_analysis": [
            {
                "hazard": "string",
                "severity": "string (high|medium|low)",
                "mitigation": "string"
            }
        ]
    }

    system = (
        "You are the Spatial Agent for VisionSpace AI, an AI-powered interior planning and space optimization platform. "
        "Your task is to analyze room geometries, clearances, accessibility paths, free space, and safety to perform layout reasoning."
    )
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Room type: **{room_type}**\n"
            f"Objects:\n{object_summary}\n"
            f"Doors:\n{door_summary}\n"
            f"Windows:\n{window_summary}\n"
            f"Proposed additions:\n{prop_summary}"
        )),
        _section("COMPUTED GEOMETRIC REPORT", geom_context),
        _section("TASK", (
            "Analyse the spatial layout of the room using the geometric report as your ground truth. "
            "Evaluate flow, balance, clearances, functional zoning, and calculate scores. "
            "For proposed additions, validate if they fit. Propose actionable recommendations "
            "to optimize furniture placement and preserve fire exits and emergency paths."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 3. Design Agent — Style & Recommendation
# ---------------------------------------------------------------------------

def design_recommendation_prompt(
    objects: list[dict],
    spatial_analysis: dict,
    room_type: str = "living_room",
    style_preference: str = "modern",
    budget_tier: str = "mid_range",
) -> dict[str, str]:
    """Prompt for the Design Agent to recommend design improvements.

    Args:
        objects:          Detected objects.
        spatial_analysis: Output from the Spatial Agent.
        room_type:        Room category.
        style_preference: Desired interior design style.
        budget_tier:      One of: budget / mid_range / premium / luxury.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    object_summary = _summarise_objects(objects)
    layout_summary = _summarise_dict(spatial_analysis, max_keys=6)
    schema = {
        "design_plan": {
            "current_style":       "string",
            "target_style":        "string",
            "style_gap":           "string",
            "colour_palette": {
                "primary":   "string",
                "secondary": "string",
                "accent":    "string",
                "neutral":   "string",
            },
            "changes": [
                {
                    "priority":     "int 1-5 (1=highest)",
                    "type":         "string (replace|add|remove|reposition|refinish)",
                    "target":       "string (object ID or zone)",
                    "description":  "string",
                    "reason":       "string",
                    "impact":       "string (high|medium|low)",
                }
            ],
            "mood_board_keywords": ["string"],
            "estimated_impact":    "string (high|medium|low)",
        }
    }

    system = _expert_system_prompt()
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Room type: **{room_type}** | Style goal: **{style_preference}** | "
            f"Budget tier: **{budget_tier}**\n"
            f"Objects:\n{object_summary}\n"
            f"Layout issues:\n{layout_summary}"
        )),
        _section("TASK", (
            f"Create a comprehensive design improvement plan targeting the "
            f"**{style_preference}** aesthetic within a **{budget_tier}** budget. "
            "Prioritise changes by impact. For each change, specify whether to "
            "replace, add, remove, reposition, or refinish an element. Include "
            "a recommended colour palette and mood-board keywords."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 4. Commerce Agent — Product Search & Ranking
# ---------------------------------------------------------------------------

def commerce_search_prompt(
    design_changes: list[dict],
    budget_tier: str = "mid_range",
    style: str = "modern",
    max_results_per_item: int = 3,
) -> dict[str, str]:
    """Prompt for the Commerce Agent to generate product search queries.

    Args:
        design_changes:      List of design change dicts from Design Agent.
        budget_tier:         Budget tier for price filtering.
        style:               Interior style to guide product selection.
        max_results_per_item: Max products to suggest per change item.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    changes_text = "\n".join(
        f"  {i+1}. [{c.get('type','?').upper()}] {c.get('description','?')}"
        for i, c in enumerate(design_changes[:15])
    )
    schema = {
        "product_searches": [
            {
                "change_index":   "int (1-indexed)",
                "search_queries": ["string (2-5 word product search queries)"],
                "category":       "string",
                "price_range":    {"min": "int USD", "max": "int USD"},
                "style_keywords": ["string"],
                "avoid_keywords": ["string"],
            }
        ]
    }

    system = _expert_system_prompt()
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Style: **{style}** | Budget: **{budget_tier}**\n"
            f"Design changes to source products for:\n{changes_text}"
        )),
        _section("TASK", (
            "For each design change, generate targeted product search queries "
            f"that will find suitable replacements. Suggest up to "
            f"{max_results_per_item} search queries per item. Include "
            "style-appropriate keywords and exclude terms that don't fit the aesthetic."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 5. Budget Agent — Cost Estimation & Breakdown
# ---------------------------------------------------------------------------

def budget_estimation_prompt(
    design_changes: list[dict],
    products: list[dict],
    total_budget: float,
    currency: str = "USD",
) -> dict[str, str]:
    """Prompt for the Budget Agent to allocate costs and prioritise spending.

    Args:
        design_changes: Design change list.
        products:       Product candidates from Commerce Agent.
        total_budget:   User's stated total budget.
        currency:       Currency code (default USD).

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    changes_text = _summarise_list(design_changes, max_items=20)
    schema = {
        "budget_plan": {
            "total_budget":    f"float {currency}",
            "allocated_total": f"float {currency}",
            "contingency":     f"float {currency} (10% recommended)",
            "items": [
                {
                    "change_index":  "int",
                    "description":   "string",
                    "priority":      "int 1-5",
                    "estimated_cost": f"float {currency}",
                    "product_name":  "string (recommended product)",
                    "retailer":      "string",
                    "url_hint":      "string (search term)",
                    "phase":         "int (1=immediate, 2=soon, 3=optional)",
                }
            ],
            "phases": {
                "phase_1_total": f"float {currency}",
                "phase_2_total": f"float {currency}",
                "phase_3_total": f"float {currency}",
            },
            "savings_tips": ["string"],
            "splurge_items": ["string (items worth spending more on)"],
        }
    }

    system = _expert_system_prompt()
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Total budget: **{currency} {total_budget:,.0f}**\n"
            f"Number of changes: {len(design_changes)}\n"
            f"Changes:\n{changes_text}"
        )),
        _section("TASK", (
            f"Allocate the {currency} {total_budget:,.0f} budget across all design "
            "changes. Group items into phases (1=immediate, 2=soon, 3=optional). "
            "Reserve 10% as contingency. Identify items where quality matters most "
            "(splurge) and where budget alternatives work fine (save). "
            "Provide 3-5 money-saving tips."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 6. Explain Agent — Natural Language Summary
# ---------------------------------------------------------------------------

def explain_design_prompt(
    vision_output: dict,
    spatial_output: dict,
    design_plan: dict,
    audience: str = "homeowner",
    tone: str = "warm_professional",
) -> dict[str, str]:
    """Prompt for the Explain Agent to produce a human-readable design summary.

    Args:
        vision_output:  Raw Vision Agent output.
        spatial_output: Spatial Agent layout analysis.
        design_plan:    Design Agent recommendations.
        audience:       Target reader (homeowner|interior_designer|real_estate_agent).
        tone:           Writing tone (warm_professional|technical|casual).

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    vision_summary = _summarise_dict(vision_output, max_keys=5)
    spatial_summary = _summarise_dict(spatial_output, max_keys=5)
    design_summary = _summarise_dict(design_plan, max_keys=6)

    schema = {
        "explanation": {
            "headline":         "string (compelling one-liner)",
            "current_state":    "string (2-3 sentences about existing room)",
            "key_issues":       ["string (brief issue descriptions)"],
            "transformation":   "string (3-4 sentences on the proposed transformation)",
            "top_3_changes":    ["string (action-oriented, specific)"],
            "style_description": "string (evocative style narrative)",
            "before_after":     "string (contrast sentence)",
            "call_to_action":   "string (encouraging closing statement)",
        }
    }

    system = (
        f"You are VisionSpace AI, an expert interior design communicator. "
        f"Your audience is a **{audience}**. Use a **{tone}** tone. "
        f"Write clearly, engagingly, and avoid jargon unless the audience is a professional."
    )
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Vision analysis highlights:\n{vision_summary}\n\n"
            f"Spatial analysis highlights:\n{spatial_summary}\n\n"
            f"Design plan highlights:\n{design_summary}"
        )),
        _section("TASK", (
            "Write a compelling, human-readable summary of the room analysis and "
            "design recommendations. Use vivid but concise language. "
            "Make the homeowner excited about the transformation."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 7. Object Enrichment Prompt
# ---------------------------------------------------------------------------

def object_enrichment_prompt(
    label: str,
    colour: str,
    material: str,
    room_type: str,
    style: str,
) -> dict[str, str]:
    """Prompt to enrich a single detected object with design metadata.

    Args:
        label:     Object label (e.g. "sofa").
        colour:    Detected colour string.
        material:  Detected material string.
        room_type: Room context.
        style:     Current or target design style.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    schema = {
        "enriched_object": {
            "label":              label,
            "style_compatibility": ["string (compatible styles)"],
            "typical_dimensions": {"width_cm": "int", "depth_cm": "int", "height_cm": "int"},
            "material_quality":   "string (budget|mid|premium|luxury)",
            "age_estimate":       "string (new|1-5 years|5-10 years|vintage|antique)",
            "replacement_priority": "int 1-5 (1=replace soon, 5=keep)",
            "keep_or_replace":    "string (keep|replace|refinish|reupholster)",
            "reason":             "string",
            "upgrade_options":    ["string (upgrade path descriptions)"],
            "accessory_pairings": ["string (complementary items)"],
        }
    }

    system = _expert_system_prompt()
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Object: **{label}** | Colour: {colour} | Material: {material}\n"
            f"Room: {room_type} | Style target: {style}"
        )),
        _section("TASK", (
            f"Provide expert design metadata for this {label}. Assess its style "
            "compatibility, estimate its dimensions and quality tier, and recommend "
            "whether to keep, replace, or refinish it. Suggest accessories that pair well."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# 8. Visualisation Prompt (Inpainting Direction)
# ---------------------------------------------------------------------------

def visualisation_prompt(
    design_changes: list[dict],
    style: str,
    colour_palette: dict,
) -> dict[str, str]:
    """Prompt to generate an inpainting / image-generation direction string.

    This is fed to the Visualisation Agent which constructs prompts for
    Stable Diffusion or similar generative models.

    Returns:
        Dict with ``"system"`` and ``"user"`` prompt strings.
    """
    changes_text = "; ".join(
        c.get("description", "") for c in design_changes[:8]
    )
    palette_text = ", ".join(f"{k}: {v}" for k, v in colour_palette.items())
    schema = {
        "visualisation": {
            "positive_prompt":  "string (SD-style prompt, comma-separated)",
            "negative_prompt":  "string (SD-style negative prompt)",
            "style_tokens":     ["string"],
            "colour_tokens":    ["string"],
            "inpaint_regions": [
                {
                    "object_id":     "string",
                    "replacement":   "string",
                    "prompt_suffix": "string",
                }
            ],
            "guidance_scale":   "float (7.5-15.0 recommended)",
            "steps":            "int (20-50 recommended)",
        }
    }

    system = (
        "You are an expert in prompt engineering for interior design image generation. "
        "You create precise, style-accurate prompts for Stable Diffusion models."
    )
    user = "\n\n".join([
        _section("CONTEXT", (
            f"Style: **{style}** | Colour palette: {palette_text}\n"
            f"Design changes: {changes_text}"
        )),
        _section("TASK", (
            "Generate a detailed Stable Diffusion prompt for visualising the "
            "proposed room transformation. Include positive and negative prompts, "
            "style tokens, colour tokens, and per-object inpainting instructions."
        )),
        _section("OUTPUT FORMAT", json.dumps(schema, indent=2)),
        _JSON_REMINDER,
    ])

    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _expert_system_prompt() -> str:
    return (
        f"{_EXPERT_IDENTITY}\n\n"
        "CRITICAL SYSTEM RULES:\n"
        "1. DOOR AND WINDOW DETECTION: You must identify every visible door and window in the scene. "
        "Do not ignore doors, glass doors, sliding doors, windows, skylights, or archways. You MUST populate "
        "both the 'doors' and 'windows' lists in the JSON structure with their estimated dimensions and 2D bounding boxes.\n"
        "2. SCENE BOUNDARIES & OBJECTS: Detect all core furniture items. Ensure that for each item "
        "you provide high-confidence labels, styles, materials, colors, and estimated dimensions.\n"
        "3. SPATIAL POSITIONING & ROTATION: You must estimate relative physical room coordinates "
        "under 'room_position' (x_cm, y_cm relative to the room floor corner) and 'rotation_deg' (estimated rotation in degrees, e.g. 0, 90, 180, 270) based on room cues.\n"
        "4. MULTI-VIEW CONSISTENCY: If analyzing multiple viewpoint images of the same room, understand "
        "that the same physical object will appear from different camera angles. Ensure that when you identify objects, "
        "you represent the physical room's layout. Keep description details, colors, styles, and IDs consistent "
        "across images so the post-processing engine can merge them cleanly into one canonical object."
    )


def _summarise_objects(objects: list[dict], max_items: int = 20) -> str:
    lines = []
    for i, obj in enumerate(objects[:max_items]):
        label = obj.get("label", "?")
        colour = obj.get("colour", "")
        bbox = obj.get("bbox", {})
        lines.append(f"  [{i+1}] {label} ({colour}) bbox={bbox}")
    if len(objects) > max_items:
        lines.append(f"  … and {len(objects) - max_items} more objects")
    return "\n".join(lines) if lines else "  (none)"


def _summarise_dict(d: dict, max_keys: int = 6) -> str:
    if not d:
        return "  (empty)"
    lines = []
    for i, (k, v) in enumerate(d.items()):
        if i >= max_keys:
            lines.append(f"  … ({len(d) - max_keys} more keys)")
            break
        if isinstance(v, (list, dict)):
            v_str = json.dumps(v)[:80] + ("…" if len(json.dumps(v)) > 80 else "")
        else:
            v_str = str(v)
        lines.append(f"  {k}: {v_str}")
    return "\n".join(lines)


def _summarise_list(items: list, max_items: int = 15) -> str:
    lines = []
    for i, item in enumerate(items[:max_items]):
        if isinstance(item, dict):
            desc = item.get("description", str(item)[:60])
        else:
            desc = str(item)[:60]
        lines.append(f"  [{i+1}] {desc}")
    if len(items) > max_items:
        lines.append(f"  … and {len(items) - max_items} more")
    return "\n".join(lines) if lines else "  (none)"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "vision_analysis_prompt",
    "spatial_analysis_prompt",
    "design_recommendation_prompt",
    "commerce_search_prompt",
    "budget_estimation_prompt",
    "explain_design_prompt",
    "object_enrichment_prompt",
    "visualisation_prompt",
]
