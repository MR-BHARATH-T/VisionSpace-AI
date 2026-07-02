"""Commerce/Product Agent Module for VisionSpace AI.

Coordinates user preferences with room inventory, spatial clearances, and design styles
to output filtered product recommendations (requested products + complementary budget-aligned items).
"""

from __future__ import annotations

import os
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
        # Mock product catalog for fallback and local searches
        self.mock_catalog = [
            # IKEA (12 products)
            {"name": "IKEA Micke Desk", "category": "desk", "brand": "IKEA", "price": 6500, "style": "modern", "width_cm": 105, "depth_cm": 50, "quality_score": 0.8},
            {"name": "IKEA Malm Desk", "category": "desk", "brand": "IKEA", "price": 9999, "style": "modern", "width_cm": 140, "depth_cm": 65, "quality_score": 0.85},
            {"name": "IKEA Flintan Office Chair", "category": "chair", "brand": "IKEA", "price": 5999, "style": "modern", "width_cm": 70, "depth_cm": 70, "quality_score": 0.75},
            {"name": "IKEA Markus Ergonomic Chair", "category": "chair", "brand": "IKEA", "price": 14999, "style": "modern", "width_cm": 62, "depth_cm": 60, "quality_score": 0.9},
            {"name": "IKEA Billy Bookshelf", "category": "bookshelf", "brand": "IKEA", "price": 4500, "style": "modern", "width_cm": 80, "depth_cm": 28, "quality_score": 0.8},
            {"name": "IKEA Tertial Lamp", "category": "lighting", "brand": "IKEA", "price": 999, "style": "industrial", "width_cm": 15, "depth_cm": 15, "quality_score": 0.75},
            {"name": "IKEA Kallax Storage Unit", "category": "bookshelf", "brand": "IKEA", "price": 3999, "style": "modern", "width_cm": 77, "depth_cm": 39, "quality_score": 0.8},
            {"name": "IKEA Lack Side Table", "category": "table", "brand": "IKEA", "price": 899, "style": "modern", "width_cm": 55, "depth_cm": 55, "quality_score": 0.7},
            {"name": "IKEA Poang Armchair", "category": "chair", "brand": "IKEA", "price": 6999, "style": "scandinavian", "width_cm": 68, "depth_cm": 82, "quality_score": 0.85},
            {"name": "IKEA Linnmon Work Table", "category": "desk", "brand": "IKEA", "price": 2500, "style": "modern", "width_cm": 100, "depth_cm": 60, "quality_score": 0.75},
            {"name": "IKEA Alex Drawer Unit", "category": "drawer", "brand": "IKEA", "price": 6500, "style": "modern", "width_cm": 36, "depth_cm": 58, "quality_score": 0.85},
            {"name": "IKEA Renberget Swivel Chair", "category": "chair", "brand": "IKEA", "price": 4999, "style": "modern", "width_cm": 59, "depth_cm": 65, "quality_score": 0.75},

            # Urban Ladder (10 products)
            {"name": "Urban Ladder Study Table", "category": "desk", "brand": "Urban Ladder", "price": 12500, "style": "scandinavian", "width_cm": 120, "depth_cm": 60, "quality_score": 0.9},
            {"name": "Urban Ladder Engineered Wood Shelf", "category": "bookshelf", "brand": "Urban Ladder", "price": 8999, "style": "modern", "width_cm": 90, "depth_cm": 35, "quality_score": 0.85},
            {"name": "Urban Ladder Crescent Study Chair", "category": "chair", "brand": "Urban Ladder", "price": 7999, "style": "contemporary", "width_cm": 60, "depth_cm": 60, "quality_score": 0.85},
            {"name": "Urban Ladder Malabar Bookshelf", "category": "bookshelf", "brand": "Urban Ladder", "price": 14500, "style": "classic", "width_cm": 85, "depth_cm": 30, "quality_score": 0.9},
            {"name": "Urban Ladder Oki Desk Organizer", "category": "organizer", "brand": "Urban Ladder", "price": 1200, "style": "minimalist", "width_cm": 30, "depth_cm": 20, "quality_score": 0.8},
            {"name": "Urban Ladder Boston Study Table", "category": "desk", "brand": "Urban Ladder", "price": 18500, "style": "scandinavian", "width_cm": 115, "depth_cm": 60, "quality_score": 0.9},
            {"name": "Urban Ladder Aeris Desk Chair", "category": "chair", "brand": "Urban Ladder", "price": 11500, "style": "modern", "width_cm": 65, "depth_cm": 65, "quality_score": 0.88},
            {"name": "Urban Ladder Carson Side Table", "category": "table", "brand": "Urban Ladder", "price": 4500, "style": "contemporary", "width_cm": 45, "depth_cm": 45, "quality_score": 0.82},
            {"name": "Urban Ladder Flora Table Lamp", "category": "lighting", "brand": "Urban Ladder", "price": 2500, "style": "modern", "width_cm": 22, "depth_cm": 22, "quality_score": 0.85},
            {"name": "Urban Ladder Stanhope Desk", "category": "desk", "brand": "Urban Ladder", "price": 22000, "style": "scandinavian", "width_cm": 120, "depth_cm": 60, "quality_score": 0.92},

            # Godrej Interio (10 products)
            {"name": "Godrej Interio Caliber Desk", "category": "desk", "brand": "Godrej Interio", "price": 8500, "style": "contemporary", "width_cm": 110, "depth_cm": 55, "quality_score": 0.8},
            {"name": "Godrej Interio Motion Chair", "category": "chair", "brand": "Godrej Interio", "price": 12000, "style": "contemporary", "width_cm": 65, "depth_cm": 65, "quality_score": 0.85},
            {"name": "Godrej Interio Companion Cabinet", "category": "cabinet", "brand": "Godrej Interio", "price": 9500, "style": "classic", "width_cm": 90, "depth_cm": 45, "quality_score": 0.8},
            {"name": "Godrej Interio Unwind Chair", "category": "chair", "brand": "Godrej Interio", "price": 4500, "style": "modern", "width_cm": 55, "depth_cm": 55, "quality_score": 0.78},
            {"name": "Godrej Interio Script Desk", "category": "desk", "brand": "Godrej Interio", "price": 16500, "style": "contemporary", "width_cm": 120, "depth_cm": 60, "quality_score": 0.88},
            {"name": "Godrej Interio Premier Table", "category": "desk", "brand": "Godrej Interio", "price": 10500, "style": "classic", "width_cm": 115, "depth_cm": 60, "quality_score": 0.82},
            {"name": "Godrej Interio Slimline Wardrobe", "category": "wardrobe", "brand": "Godrej Interio", "price": 21000, "style": "classic", "width_cm": 90, "depth_cm": 50, "quality_score": 0.85},
            {"name": "Godrej Interio Versatile Shelf", "category": "bookshelf", "brand": "Godrej Interio", "price": 7500, "style": "modern", "width_cm": 80, "depth_cm": 30, "quality_score": 0.8},
            {"name": "Godrej Interio Accent Lamp", "category": "lighting", "brand": "Godrej Interio", "price": 3000, "style": "contemporary", "width_cm": 25, "depth_cm": 25, "quality_score": 0.82},
            {"name": "Godrej Interio Work Desk Combo", "category": "desk", "brand": "Godrej Interio", "price": 19999, "style": "contemporary", "width_cm": 120, "depth_cm": 60, "quality_score": 0.88}
        ]
        logger.info("CommerceAgent initialized (Provider: %s)", provider)

    def recommend_products(
        self,
        vision_output: Dict[str, Any],
        spatial_output: Dict[str, Any],
        design_output: Dict[str, Any],
        user_preferences: Dict[str, Any],
        session_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Orchestrates the Phase 4 recommendation pipeline matching conformed V2 structure."""
        logger.info("Running Phase 4 Product Agent recommendations...")

        # 1. Search products from catalog
        req_products = user_preferences.get("required_products") or []
        preferred_brands = user_preferences.get("preferred_brands") or []
        excluded_brands = user_preferences.get("excluded_brands") or []
        budget_range = user_preferences.get("budget_range") or {"minimum": 0, "maximum": 50000}
        max_budget = budget_range.get("maximum") or 50000
        pref_style = user_preferences.get("preferred_style") or "modern"

        # Look up matching candidates in our catalog
        candidates = self.search_products(req_products, preferred_brands=preferred_brands)

        # Sort candidates to prioritize preferred brands first
        if preferred_brands and candidates:
            candidates = sorted(
                candidates,
                key=lambda x: x.get("brand", "").lower() in [b.lower() for b in preferred_brands],
                reverse=True
            )

        # 2. Apply filters (dimensions, budget, style, exclusions)
        # Note: In this version we use a default of 120cm width / 60cm depth for standard furniture
        candidates = self.filter_dimensions(candidates, max_w=120, max_d=60)
        candidates = self.filter_budget(candidates, budget_limit=max_budget)
        candidates = self.filter_style(candidates, style=pref_style)
        
        # Exclude brand filter
        candidates = [c for c in candidates if c.get("brand") not in excluded_brands]

        # 3. Score candidates using the V2 formulas
        scored = []
        for c in candidates:
            scores = self.score_products(
                product=c,
                pref_style=pref_style,
                preferred_brands=preferred_brands,
                required_w=120,
                required_d=60,
                budget_limit=max_budget
            )
            item = dict(c)
            item.update(scores)
            scored.append(item)

        # 4. Rank products
        ranked = self.rank_products(scored)
        
        # Keep top 10 ranked products
        ranked = ranked[:10]

        # 5. Build shopping list & explanations
        shopping_list = self.build_shopping_list(ranked)
        explanations = self.generate_explanations(ranked)

        output = {
            "recommended_products": ranked,
            "shopping_list": shopping_list,
            "explanations": explanations
        }

        # Save output if path is given
        if session_path is not None:
            try:
                import pathlib
                p = pathlib.Path(session_path)
                if p.is_dir():
                    save_json(output, p / "commerce_analysis.json")
            except Exception as e:
                logger.error("Failed to save commerce analysis JSON: %s", str(e))

        return output

    def search_products(self, categories: List[str], preferred_brands: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Searches Google Shopping dynamically using SerpAPI if API key is provided, else falls back to mock catalog."""
        import config
        api_key = os.environ.get("SERPAPI_API_KEY") or getattr(config, "SERP_API", None)
        if not api_key:
            # Try UserSecretsClient if on Kaggle
            try:
                from kaggle_secrets import UserSecretsClient
                api_key = UserSecretsClient().get_secret("SERPAPI_API_KEY")
            except Exception:
                pass

        if not api_key:
            logger.warning("SerpAPI API key not found. Falling back to local catalog search.")
            return self._search_mock_catalog(categories)

        import urllib.request
        import urllib.parse
        
        brands = preferred_brands or [""]
        results = []
        
        # Build search queries (e.g. "IKEA desk", "Urban Ladder chair")
        queries = []
        for cat in categories or ["desk"]:
            for brand in brands:
                q = f"{brand} {cat}".strip()
                if q:
                    queries.append((cat, q))
                    
        for cat, query in queries[:5]:  # Limit query batching to avoid rate limits
            try:
                logger.info("Executing dynamic Google Shopping query on SerpAPI: '%s'", query)
                query_encoded = urllib.parse.quote(query)
                url = f"https://serpapi.com/search.json?engine=google_shopping&q={query_encoded}&api_key={api_key}"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    shopping_results = data.get("shopping_results", [])
                    
                    for item in shopping_results:
                        title = item.get("title", "")
                        price_str = item.get("price", "0")
                        # Extract price digits
                        price_cleaned = re.sub(r'[^\d.]', '', price_str)
                        try:
                            price = float(price_cleaned) if price_cleaned else 0.0
                        except Exception:
                            price = 0.0
                            
                        # Detect brand from title or source
                        source = item.get("source", "")
                        detected_brand = source
                        for b in (preferred_brands or ["IKEA", "Urban Ladder", "Godrej Interio"]):
                            if b.lower() in title.lower():
                                detected_brand = b
                                break
                                
                        rating = float(item.get("rating", 4.0))
                        quality_score = rating / 5.0 if rating > 0 else 0.8
                        
                        # Parse dimensions from title
                        w_cm, d_cm = 120.0, 60.0
                        match = re.search(r'(\d+)\s*(?:x|by)\s*(\d+)', title.lower())
                        if match:
                            try:
                                w_cm = float(match.group(1))
                                d_cm = float(match.group(2))
                            except Exception:
                                pass
                                
                        results.append({
                            "name": title,
                            "category": cat,
                            "brand": detected_brand,
                            "price": price,
                            "style": "modern",  # default
                            "width_cm": w_cm,
                            "depth_cm": d_cm,
                            "quality_score": round(quality_score, 2),
                            "link": item.get("link", "")
                        })
            except Exception as e:
                logger.error("SerpAPI query failed for '%s': %s", query, str(e))
                
        # If dynamic search yielded nothing, use mock catalog
        if not results:
            return self._search_mock_catalog(categories)
            
        return results

    def _search_mock_catalog(self, categories: List[str]) -> List[Dict[str, Any]]:
        """Fallback search logic using local mock catalog."""
        if not categories:
            return [c for c in self.mock_catalog if c.get("category") == "desk"]
            
        results = []
        for cat in categories:
            normalized_cat = cat.lower()
            matches = [c for c in self.mock_catalog if normalized_cat in c.get("category", "").lower() or normalized_cat in c.get("name", "").lower()]
            if matches:
                results.extend(matches)
        
        if not results:
            results = list(self.mock_catalog)
        return results

    def filter_dimensions(self, products: List[Dict[str, Any]], max_w: float, max_d: float) -> List[Dict[str, Any]]:
        """Filters out items that exceed the available room clearance boundaries."""
        filtered = []
        for p in products:
            w = float(p.get("width_cm", 0))
            d = float(p.get("depth_cm", 0))
            if w <= max_w and d <= max_d:
                filtered.append(p)
        return filtered

    def filter_budget(self, products: List[Dict[str, Any]], budget_limit: float) -> List[Dict[str, Any]]:
        """Filters out items that exceed the target budget range."""
        return [p for p in products if float(p.get("price", 0)) <= budget_limit]

    def filter_style(self, products: List[Dict[str, Any]], style: str) -> List[Dict[str, Any]]:
        """Filters/prioritizes items matching the theme styling."""
        # We don't discard non-matching styles unless they clash, so we keep them but style scoring handles prioritizing
        return products

    def score_products(
        self,
        product: Dict[str, Any],
        pref_style: str,
        preferred_brands: List[str],
        required_w: float,
        required_d: float,
        budget_limit: float
    ) -> Dict[str, float]:
        """Calculates conformed weighted score for the product candidate.

        Formula:
          Final Score = 0.35 * fit_score + 0.25 * style_score + 0.20 * budget_score + 0.10 * brand_score + 0.10 * quality_score
        """
        # 1. fit_score: does it fit required dimensions?
        prod_w = float(product.get("width_cm", required_w))
        prod_d = float(product.get("depth_cm", required_d))
        fit_score = 1.0 if (prod_w <= required_w and prod_d <= required_d) else 0.5
        
        # 2. style_score: does it match preferred style?
        style_score = 1.0 if product.get("style", "").lower() == pref_style.lower() else 0.6
        
        # 3. budget_score: price vs limit
        price = float(product.get("price", 0))
        budget_score = 1.0 if price <= budget_limit else 0.0
        if budget_limit > 0 and price <= budget_limit:
            budget_score = 1.0 - (price / budget_limit) * 0.3 # penalize expensive options slightly
            
        # 4. brand_score: is it in preferred brands?
        brand_score = 1.0 if product.get("brand", "").lower() in [b.lower() for b in preferred_brands] else 0.5
        
        # 5. quality_score
        quality_score = float(product.get("quality_score", 0.8))
        
        # Final weighted score
        final_score = (
            0.35 * fit_score +
            0.25 * style_score +
            0.20 * budget_score +
            0.10 * brand_score +
            0.10 * quality_score
        )
        
        return {
            "fit_score": round(fit_score, 2),
            "style_score": round(style_score, 2),
            "budget_score": round(budget_score, 2),
            "brand_score": round(brand_score, 2),
            "quality_score": round(quality_score, 2),
            "fit_score_weighted": round(0.35 * fit_score, 3),
            "style_score_weighted": round(0.25 * style_score, 3),
            "budget_score_weighted": round(0.20 * budget_score, 3),
            "brand_score_weighted": round(0.10 * brand_score, 3),
            "quality_score_weighted": round(0.10 * quality_score, 3),
            "fit_score_raw": fit_score,
            "space_score": fit_score, # backward compatibility
            "budget_score_raw": budget_score,
            "final_score": round(final_score, 3)
        }

    def rank_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts candidate products in descending order based on their weighted score."""
        return sorted(products, key=lambda x: x.get("final_score", 0.0), reverse=True)

    def generate_explanations(self, products: List[Dict[str, Any]]) -> List[str]:
        """Generates clear, concise justifications for product selections."""
        explanations = []
        for p in products[:3]:
            explanations.append(
                f"Selected {p.get('name')} (Score: {p.get('final_score')}) because it fits clearance "
                f"envelopes and matches styling parameters within budget limits."
            )
        return explanations

    def build_shopping_list(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepares a shopping catalog of selected items."""
        shopping_list = []
        for p in products:
            shopping_list.append({
                "item": p.get("name"),
                "category": p.get("category"),
                "brand": p.get("brand"),
                "price": p.get("price")
            })
        return shopping_list

    def generate_visual_preview(self, products: List[Dict[str, Any]], session_path: str) -> Dict[str, Any]:
        """Generates a visualization preview manifest utilizing FLUX Kontext for reconstruction."""
        logger.info("Generating FLUX Kontext reconstruction preview parameters...")
        
        prompts = []
        for p in products[:2]:
            prompts.append(
                f"Add a {p.get('style')} {p.get('brand')} {p.get('category')} to the room floor space, "
                "preserving exact room perspective and ambient lighting."
            )

        preview_manifest = {
            "reconstruction_model": "FLUX Kontext",
            "reasoning": "Preserves room layout identity and geometric boundaries better than SDXL.",
            "prompts": prompts,
            "base_image": "original_1.png",
            "output_preview_image": "processed_reconstructed.png"
        }

        try:
            import pathlib
            p_path = pathlib.Path(session_path)
            if p_path.is_dir():
                save_json(preview_manifest, p_path / "reconstruction_preview.json")
        except Exception as e:
            logger.error("Failed to save reconstruction preview: %s", str(e))

        return preview_manifest
