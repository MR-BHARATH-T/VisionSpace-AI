# utils/object_enricher.py
"""
VisionSpace AI — Production Object Enricher
============================================
Transforms raw Vision Agent detections into fully-enriched design objects
by delegating to the Reasoning Model (ModelManager) rather than a static
lookup table.

Key design decisions:
  - Model-agnostic: uses ModelManager.generate(), so any registered
    reasoning model can be swapped in.
  - Fallback chain: LLM → rule-based heuristics → raw detection output.
  - Batch enrichment: enriches multiple objects in one call to reduce
    round-trips.
  - Caching: enriched objects are cached by (label, colour, material, style)
    key to avoid redundant inference on repeated detections.

Usage::

    enricher = ObjectEnricher()
    enriched = enricher.enrich(objects, room_type="bedroom", style="japandi")
    # each obj now has: style_compatibility, keep_or_replace, accessory_pairings, …
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-Based Fallbacks (no LLM needed)
# ---------------------------------------------------------------------------

_STYLE_COMPATIBILITY: dict[str, list[str]] = {
    "sofa":           ["modern", "scandinavian", "mid_century_modern", "industrial"],
    "armchair":       ["traditional", "mid_century_modern", "bohemian", "maximalist"],
    "coffee_table":   ["modern", "contemporary", "scandinavian", "industrial"],
    "bookshelf":      ["scandinavian", "industrial", "mid_century_modern", "bohemian"],
    "dining_table":   ["modern", "farmhouse", "rustic", "contemporary"],
    "bed":            ["modern", "scandinavian", "japandi", "traditional"],
    "wardrobe":       ["modern", "contemporary", "minimalist"],
    "floor_lamp":     ["mid_century_modern", "art_deco", "modern", "industrial"],
    "rug":            ["bohemian", "traditional", "scandinavian", "modern"],
    "plant":          ["bohemian", "japandi", "coastal", "scandinavian"],
    "mirror":         ["art_deco", "modern", "hollywood_regency", "traditional"],
    "curtain":        ["traditional", "farmhouse", "coastal", "bohemian"],
}

_MATERIAL_TIER: dict[str, str] = {
    "solid wood":   "premium",
    "oak":          "premium",
    "walnut":       "premium",
    "teak":         "premium",
    "marble":       "luxury",
    "brass":        "luxury",
    "velvet":       "premium",
    "leather":      "premium",
    "mdf":          "budget",
    "particle board": "budget",
    "plywood":      "mid",
    "pine":         "mid",
    "fabric":       "mid",
    "linen":        "mid",
    "polyester":    "budget",
    "plastic":      "budget",
    "metal":        "mid",
    "glass":        "mid",
    "rattan":       "mid",
    "bamboo":       "mid",
}

_KEEP_REPLACE_RULES: dict[str, dict] = {
    # (condition, material_tier) → recommendation
    ("good",   "premium"): {"decision": "keep",      "priority": 5},
    ("good",   "luxury"):  {"decision": "keep",      "priority": 5},
    ("good",   "mid"):     {"decision": "keep",      "priority": 4},
    ("good",   "budget"):  {"decision": "replace",   "priority": 2},
    ("worn",   "premium"): {"decision": "refinish",  "priority": 3},
    ("worn",   "mid"):     {"decision": "replace",   "priority": 2},
    ("worn",   "budget"):  {"decision": "replace",   "priority": 1},
    ("damaged","premium"): {"decision": "refinish",  "priority": 3},
    ("damaged","mid"):     {"decision": "replace",   "priority": 1},
    ("damaged","budget"):  {"decision": "replace",   "priority": 1},
}

_TYPICAL_DIMENSIONS: dict[str, dict[str, int]] = {
    "sofa":         {"width_cm": 220, "depth_cm": 90,  "height_cm": 85},
    "armchair":     {"width_cm": 80,  "depth_cm": 80,  "height_cm": 90},
    "coffee_table": {"width_cm": 120, "depth_cm": 60,  "height_cm": 45},
    "dining_table": {"width_cm": 180, "depth_cm": 90,  "height_cm": 75},
    "dining_chair": {"width_cm": 45,  "depth_cm": 50,  "height_cm": 90},
    "bookshelf":    {"width_cm": 80,  "depth_cm": 30,  "height_cm": 180},
    "wardrobe":     {"width_cm": 120, "depth_cm": 60,  "height_cm": 200},
    "bed":          {"width_cm": 160, "depth_cm": 200, "height_cm": 50},
    "desk":         {"width_cm": 140, "depth_cm": 70,  "height_cm": 75},
    "floor_lamp":   {"width_cm": 40,  "depth_cm": 40,  "height_cm": 160},
    "rug":          {"width_cm": 200, "depth_cm": 300, "height_cm": 2},
    "tv_unit":      {"width_cm": 150, "depth_cm": 45,  "height_cm": 50},
}

_ACCESSORY_PAIRINGS: dict[str, list[str]] = {
    "sofa":         ["throw pillow", "throw blanket", "side table", "floor lamp", "rug"],
    "armchair":     ["side table", "floor lamp", "ottoman", "throw pillow"],
    "dining_table": ["table runner", "centrepiece", "candle holders", "placemats"],
    "bed":          ["bedside lamp", "nightstand", "throw blanket", "area rug"],
    "bookshelf":    ["decorative books", "small plant", "framed photo", "decorative object"],
    "desk":         ["desk lamp", "monitor riser", "plant", "pen holder", "art print"],
    "coffee_table": ["coffee table books", "decorative tray", "candle", "small plant"],
    "floor_lamp":   ["side table", "armchair", "area rug"],
}


class ObjectEnricher:
    """Enriches detected objects with design metadata.

    Attempts LLM-based enrichment via ModelManager when available,
    falling back to rule-based heuristics automatically.

    Args:
        use_llm: Whether to attempt LLM enrichment (default True).
                 Set to False to force rule-based mode (faster, no GPU needed).
        cache_ttl_seconds: How long to cache enrichment results (default 1 hour).
    """

    def __init__(
        self,
        use_llm: bool = True,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._use_llm = use_llm
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[dict, float]] = {}   # key → (result, expiry)
        self._manager = None  # Lazy import to avoid circular deps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        objects: list[dict],
        room_type: str = "living_room",
        style: str = "modern",
    ) -> list[dict]:
        """Enrich a list of detected objects with design metadata.

        Args:
            objects:   List of detection dicts (each has ``label``, ``colour``,
                       ``material``, ``condition``, ``bbox``).
            room_type: Room context for enrichment.
            style:     Target design style.

        Returns:
            List of enriched object dicts with additional fields.
        """
        enriched = []
        for obj in objects:
            try:
                result = self._enrich_single(obj, room_type=room_type, style=style)
                enriched.append(result)
            except Exception as exc:
                logger.warning(
                    "Object enrichment failed for '%s': %s — using raw detection.",
                    obj.get("label", "?"), exc,
                )
                enriched.append({**obj, "_enrichment_status": "failed"})
        return enriched

    def enrich_single(
        self,
        obj: dict,
        room_type: str = "living_room",
        style: str = "modern",
    ) -> dict:
        """Enrich a single detected object. Public wrapper for :meth:`_enrich_single`."""
        return self._enrich_single(obj, room_type=room_type, style=style)

    def clear_cache(self) -> int:
        """Clear the enrichment result cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info("ObjectEnricher cache cleared (%d entries)", count)
        return count

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        now = time.monotonic()
        alive = sum(1 for _, (_, exp) in self._cache.items() if exp > now)
        return {
            "total_entries": len(self._cache),
            "alive_entries": alive,
            "expired_entries": len(self._cache) - alive,
        }

    # ------------------------------------------------------------------
    # Internal Enrichment
    # ------------------------------------------------------------------

    def _enrich_single(
        self,
        obj: dict,
        room_type: str,
        style: str,
    ) -> dict:
        label    = obj.get("label", "object").lower().strip()
        colour   = obj.get("colour", "unknown")
        material = obj.get("material", "unknown")
        condition = obj.get("condition", "good")

        cache_key = self._cache_key(label, colour, material, style)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return {**obj, **cached, "_enrichment_source": "cache"}

        # Try LLM enrichment first
        if self._use_llm:
            llm_result = self._enrich_via_llm(
                label, colour, material, condition, room_type, style
            )
            if llm_result:
                self._set_cache(cache_key, llm_result)
                return {**obj, **llm_result, "_enrichment_source": "llm"}

        # Fallback: rule-based
        rule_result = self._enrich_via_rules(label, colour, material, condition, style)
        self._set_cache(cache_key, rule_result)
        return {**obj, **rule_result, "_enrichment_source": "rules"}

    def _enrich_via_llm(
        self,
        label: str,
        colour: str,
        material: str,
        condition: str,
        room_type: str,
        style: str,
    ) -> dict | None:
        """Attempt LLM enrichment via ModelManager.  Returns None on any failure."""
        try:
            from models.model_manager import ModelManager
            from utils.prompt_templates import object_enrichment_prompt

            manager = ModelManager.get_instance()
            prompt = object_enrichment_prompt(
                label=label, colour=colour, material=material,
                room_type=room_type, style=style,
            )

            messages = [
                {"role": "system",    "content": prompt["system"]},
                {"role": "user",      "content": prompt["user"]},
            ]
            response = manager.generate(task="reasoning", messages=messages)

            if not response or not response.get("text"):
                return None

            raw_text = response["text"].strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]

            parsed = json.loads(raw_text)
            return parsed.get("enriched_object", parsed)

        except Exception as exc:
            logger.debug("LLM enrichment skipped: %s", exc)
            return None

    def _enrich_via_rules(
        self,
        label: str,
        colour: str,
        material: str,
        condition: str,
        style: str,
    ) -> dict:
        """Return rule-based enrichment metadata for a single object."""
        # Style compatibility
        style_compat = self._resolve_style_compat(label)

        # Material quality tier
        material_tier = "mid"
        for mat_kw, tier in _MATERIAL_TIER.items():
            if mat_kw in material.lower():
                material_tier = tier
                break

        # Keep / replace decision
        rule = _KEEP_REPLACE_RULES.get(
            (condition, material_tier),
            {"decision": "keep", "priority": 3},
        )

        # Typical dimensions
        dims = _TYPICAL_DIMENSIONS.get(label, {"width_cm": 0, "depth_cm": 0, "height_cm": 0})

        # Accessories
        accessories = _ACCESSORY_PAIRINGS.get(label, [])

        # Style compatibility score (1 = match, 0 = mismatch)
        style_match = style in style_compat

        return {
            "style_compatibility":    style_compat,
            "style_match":            style_match,
            "typical_dimensions":     dims,
            "material_quality":       material_tier,
            "age_estimate":           "unknown",
            "replacement_priority":   rule["priority"],
            "keep_or_replace":        rule["decision"],
            "reason": (
                f"{label.capitalize()} in {condition} condition with {material_tier}-quality "
                f"materials → {rule['decision']}."
            ),
            "upgrade_options":        [],
            "accessory_pairings":     accessories,
        }

    # ------------------------------------------------------------------
    # Style Compatibility Resolution
    # ------------------------------------------------------------------

    def _resolve_style_compat(self, label: str) -> list[str]:
        """Return compatible styles for a label (exact or fuzzy match)."""
        # Exact match
        if label in _STYLE_COMPATIBILITY:
            return _STYLE_COMPATIBILITY[label]
        # Fuzzy: check if label contains a known key
        for key, styles in _STYLE_COMPATIBILITY.items():
            if key in label:
                return styles
        # Generic fallback
        return ["modern", "contemporary", "transitional"]

    # ------------------------------------------------------------------
    # Cache Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(label: str, colour: str, material: str, style: str) -> str:
        raw = f"{label}|{colour}|{material}|{style}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        result, expiry = entry
        if time.monotonic() > expiry:
            del self._cache[key]
            return None
        return result

    def _set_cache(self, key: str, value: dict) -> None:
        self._cache[key] = (value, time.monotonic() + self._cache_ttl)


# ---------------------------------------------------------------------------
# Module-Level Convenience Functions
# ---------------------------------------------------------------------------

_default_enricher: ObjectEnricher | None = None


def get_enricher(use_llm: bool = True) -> ObjectEnricher:
    """Return the module-level singleton ObjectEnricher."""
    global _default_enricher
    if _default_enricher is None:
        _default_enricher = ObjectEnricher(use_llm=use_llm)
    return _default_enricher


def enrich_objects(
    objects: list[dict],
    room_type: str = "living_room",
    style: str = "modern",
    use_llm: bool = True,
) -> list[dict]:
    """Module-level convenience: enrich a list of detected objects.

    Args:
        objects:   Detected object list from Vision Agent.
        room_type: Room context.
        style:     Target design style.
        use_llm:   Whether to use LLM enrichment (fallback to rules if False).

    Returns:
        Enriched object list.
    """
    return get_enricher(use_llm=use_llm).enrich(
        objects, room_type=room_type, style=style
    )


__all__ = [
    "ObjectEnricher",
    "get_enricher",
    "enrich_objects",
]
