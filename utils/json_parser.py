"""JSON Parser and Validation Schemas Module for VisionSpace AI.

Defines Pydantic schemas for the agentic workflow to ensure structured validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger("VisionSpaceAI.JsonParser")


# =========================================================================
# Nested Schema Helper Definitions
# =========================================================================

class SpatialObject(BaseModel):
    """Details of a single detected object in space."""
    name: str = Field(description="Name or label of the object.")
    bounding_box: List[float] = Field(
        default=[],
        description="Bounding box coordinates, typically normalized [ymin, xmin, ymax, xmax]."
    )
    confidence: float = Field(default=1.0, description="Detection confidence score from 0 to 1.")
    size_estimate: Dict[str, float] = Field(
        default={},
        description="Estimated dimensions of the object (e.g. width, height, depth in meters)."
    )


class BudgetItem(BaseModel):
    """A single furniture/materials item in the budget estimator."""
    name: str = Field(description="Name or description of the product/material.")
    estimated_price: float = Field(description="Estimated price of the item.")
    priority: str = Field(
        default="medium",
        description="Priority of purchase, e.g. 'essential', 'recommended', 'optional'."
    )


class ProductMatch(BaseModel):
    """An e-commerce product matching a recommendation."""
    item_name: str = Field(description="E-commerce catalog item name.")
    store_url: str = Field(description="URL to the store product page.")
    price: float = Field(description="Catalog item price.")
    match_score: float = Field(default=1.0, description="Similarity match score from 0 to 1.")


# =========================================================================
# Core Schemas for the 6 Agents
# =========================================================================

class VisionSchema(BaseModel):
    """Schema for the Vision Agent analyzing the room image."""
    room_type: str = Field(description="Detected room type (e.g. living room, bedroom, kitchen).")
    style: str = Field(description="Design style detected (e.g. minimalist, scandinavian, industrial).")
    lighting: str = Field(description="Lighting conditions and sources detected (e.g. bright natural, dim warm).")
    colors: List[str] = Field(description="Primary color palette detected in the space.")
    description: str = Field(description="Detailed visual analysis summary of the room.")


class SpatialSchema(BaseModel):
    """Schema for the Spatial Agent identifying objects and layout coordinates."""
    objects: List[SpatialObject] = Field(default=[], description="List of detected spatial objects.")
    dimensions_estimate: Dict[str, float] = Field(
        default={},
        description="Overall estimated room dimensions (e.g., width, length, ceiling_height in meters)."
    )
    spatial_relations: List[str] = Field(
        default=[],
        description="Descriptions of object relations (e.g., 'sofa is against the east wall')."
    )


class DesignSchema(BaseModel):
    """Schema for the Design Agent generating improvement concepts."""
    style_recommendations: List[str] = Field(description="Style upgrades or thematic recommendation list.")
    layout_changes: List[str] = Field(description="Suggested spatial reorganizations or furniture shifts.")
    furniture_additions: List[str] = Field(description="Suggested furniture additions or replacements.")
    color_palette_suggestions: List[str] = Field(description="Proposed new color palette scheme.")
    overall_concept: str = Field(description="Overall conceptual theme description.")


class BudgetSchema(BaseModel):
    """Schema for the Budget Agent listing cost estimates."""
    items: List[BudgetItem] = Field(default=[], description="Itemized budget breakdown.")
    total_estimated_cost: float = Field(description="Total summed cost of recommended design upgrades.")
    budget_status: str = Field(
        description="Statement on whether the plan fits target budget parameters (e.g., 'within budget', 'over budget')."
    )


class CommerceSchema(BaseModel):
    """Schema for the Commerce Agent sourcing matching items."""
    product_matches: List[ProductMatch] = Field(default=[], description="Sourced products from retail partners.")
    recommendations: List[str] = Field(description="General procurement advice or shopping guidance.")


class ExplainSchema(BaseModel):
    """Schema for the Explain Agent giving final justification explanations."""
    design_decisions_explanation: str = Field(description="Deep dive explanation of why design choices were made.")
    budget_justification: str = Field(description="Justification of the budget distribution.")
    next_steps: List[str] = Field(description="Recommended sequential action items for implementation.")


# =========================================================================
# Validation Helper Functions
# =========================================================================

def validate_json_schema(data: Dict[str, Any], schema: Type[BaseModel]) -> bool:
    """Validates a dictionary data structure against a given Pydantic schema class.

    Args:
        data: The JSON dictionary object to validate.
        schema: A Pydantic BaseModel class definition.

    Returns:
        True if validation succeeds, False otherwise.
    """
    if schema is None:
        return True
    try:
        if hasattr(schema, "model_validate"):
            # Pydantic V2
            schema.model_validate(data)
        elif hasattr(schema, "validate"):
            # Pydantic V1 validation helper
            schema.validate(data)
        else:
            # Direct instantiation fallback
            schema(**data)
        return True
    except Exception as e:
        logger.warning("JSON Schema validation failed for %s: %s", schema.__name__, str(e))
        return False


def extract_and_validate_json(text: str) -> Optional[Any]:
    """Extracts JSON from text, parsing markdown code blocks if necessary, and validating it."""
    import re
    text = text.strip()
    
    # Try parsing directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try matching ```json ... ``` code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try matching any curly braces block { ... } or [ ... ]
    brace_pattern = r"(\{.*?\}|\[.*?\])"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            pass

    return None


def has_empty_array(data: Any) -> bool:
    """Recursively checks if any array/list in the JSON structure is empty."""
    if isinstance(data, list):
        if not data:
            return True
        return any(has_empty_array(item) for item in data)
    elif isinstance(data, dict):
        return any(has_empty_array(val) for val in data.values())
    return False


def has_duplicate_objects(data: Any) -> bool:
    """Recursively checks if any array/list in the JSON contains duplicate items/objects."""
    if isinstance(data, list):
        seen = []
        for item in data:
            if isinstance(item, (dict, list)):
                serialized = json.dumps(item, sort_keys=True)
                if serialized in seen:
                    return True
                seen.append(serialized)
            else:
                if item in seen:
                    return True
                seen.append(item)
        return any(has_duplicate_objects(item) for item in data)
    elif isinstance(data, dict):
        return any(has_duplicate_objects(val) for val in data.values())
    return False


def validate_response_rules(
    response_text: str,
    parsed_json: Optional[Any],
    validator: Optional[Dict[str, Any]]
) -> tuple[bool, str]:
    """Validates the response text or parsed JSON against rule configurations.

    Args:
        response_text: The raw generated string response.
        parsed_json: The parsed JSON object (if JSON format requested).
        validator: A dictionary of validation rules.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if not validator:
        return True, ""

    # 1. Word count validation
    min_words = validator.get("min_words", 0)
    if min_words > 0:
        word_count = len(response_text.split())
        if word_count < min_words:
            return False, f"Response word count ({word_count}) is less than minimum words required ({min_words})."

    # If the validator has JSON-related rules but parsing failed
    requires_json = any(k in validator for k in ["required_keys", "schema", "allow_empty_arrays", "allow_duplicate_objects"])
    if requires_json and parsed_json is None:
        return False, "Response is not valid JSON, but JSON validation rules are specified."

    if parsed_json is not None:
        # 2. Required keys check
        if isinstance(parsed_json, dict):
            required_keys = validator.get("required_keys", [])
            for key in required_keys:
                if key not in parsed_json:
                    return False, f"Missing required JSON key: '{key}'."

        # 3. Schema validation (Pydantic BaseModel)
        schema_class = validator.get("schema")
        if schema_class:
            if isinstance(parsed_json, dict):
                if not validate_json_schema(parsed_json, schema_class):
                    return False, f"JSON fails validation against Pydantic schema '{schema_class.__name__}'."
            else:
                return False, f"Expected JSON object for schema validation, but got {type(parsed_json).__name__}."

        # 4. Empty arrays check
        if not validator.get("allow_empty_arrays", True):
            if has_empty_array(parsed_json):
                return False, "Response contains an empty list/array, which is forbidden."

        # 5. Duplicate objects check
        if not validator.get("allow_duplicate_objects", True):
            if has_duplicate_objects(parsed_json):
                return False, "Response contains duplicate items/objects in a list, which is forbidden."

    return True, ""
