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

def extract_direct_link(link: str) -> str:
    """Extracts direct merchant link (IKEA, Amazon, Flipkart, etc.) from Google redirect URLs."""
    if not link:
        return ""
    if "google.com/url" in link or "google.adservices.com" in link:
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(link)
            params = urllib.parse.parse_qs(parsed.query)
            for key in ["adurl", "q", "url"]:
                if key in params and params[key]:
                    return params[key][0]
        except Exception:
            pass
    return link


class CommerceAgent:
    """Production-grade CommerceAgent responsible for item curation matching V2 schemas."""

    def __init__(self, provider: str = "huggingface") -> None:
        """Initializes the CommerceAgent."""
        self.provider_name = provider
        self.text_model = TextModel(provider=provider)
        # Mock product catalog for fallback and local searches
        self.mock_catalog = [
            # IKEA (12 products)
            {"name": "IKEA Micke Desk", "category": "desk", "brand": "IKEA", "price": 6500, "style": "modern", "width_cm": 105, "depth_cm": 50, "quality_score": 0.8, "link": "https://www.ikea.com/in/en/p/micke-desk-black-brown-80354274/"},
            {"name": "IKEA Malm Desk", "category": "desk", "brand": "IKEA", "price": 9999, "style": "modern", "width_cm": 140, "depth_cm": 65, "quality_score": 0.85, "link": "https://www.ikea.com/in/en/p/malm-desk-white-10359824/"},
            {"name": "IKEA Flintan Office Chair", "category": "chair", "brand": "IKEA", "price": 5999, "style": "modern", "width_cm": 70, "depth_cm": 70, "quality_score": 0.75, "link": "https://www.ikea.com/in/en/p/flintan-office-chair-with-armrests-black-s39434057/"},
            {"name": "IKEA Markus Ergonomic Chair", "category": "chair", "brand": "IKEA", "price": 14999, "style": "modern", "width_cm": 62, "depth_cm": 60, "quality_score": 0.9, "link": "https://www.ikea.com/in/en/p/markus-office-chair-vissle-dark-grey-50261151/"},
            {"name": "IKEA Billy Bookshelf", "category": "bookshelf", "brand": "IKEA", "price": 4500, "style": "modern", "width_cm": 80, "depth_cm": 28, "quality_score": 0.8, "link": "https://www.ikea.com/in/en/p/billy-bookcase-white-80351561/"},
            {"name": "IKEA Tertial Lamp", "category": "lighting", "brand": "IKEA", "price": 999, "style": "industrial", "width_cm": 15, "depth_cm": 15, "quality_score": 0.75, "link": "https://www.ikea.com/in/en/p/tertial-work-lamp-dark-grey-50355721/"},
            {"name": "IKEA Kallax Storage Unit", "category": "bookshelf", "brand": "IKEA", "price": 3999, "style": "modern", "width_cm": 77, "depth_cm": 39, "quality_score": 0.8, "link": "https://www.ikea.com/in/en/p/kallax-shelving-unit-white-80351882/"},
            {"name": "IKEA Lack Side Table", "category": "table", "brand": "IKEA", "price": 899, "style": "modern", "width_cm": 55, "depth_cm": 55, "quality_score": 0.7, "link": "https://www.ikea.com/in/en/p/lack-side-table-white-10302064/"},
            {"name": "IKEA Poang Armchair", "category": "chair", "brand": "IKEA", "price": 6999, "style": "scandinavian", "width_cm": 68, "depth_cm": 82, "quality_score": 0.85, "link": "https://www.ikea.com/in/en/p/poaeng-armchair-glose-black-s99830501/"},
            {"name": "IKEA Linnmon Work Table", "category": "desk", "brand": "IKEA", "price": 2500, "style": "modern", "width_cm": 100, "depth_cm": 60, "quality_score": 0.75, "link": "https://www.ikea.com/in/en/p/linnmon-table-top-white-60361189/"},
            {"name": "IKEA Alex Drawer Unit", "category": "drawer", "brand": "IKEA", "price": 6500, "style": "modern", "width_cm": 36, "depth_cm": 58, "quality_score": 0.85, "link": "https://www.ikea.com/in/en/p/alex-drawer-unit-white-40473549/"},
            {"name": "IKEA Renberget Swivel Chair", "category": "chair", "brand": "IKEA", "price": 4999, "style": "modern", "width_cm": 59, "depth_cm": 65, "quality_score": 0.75, "link": "https://www.ikea.com/in/en/p/renberget-swivel-chair-bomstad-black-20339414/"},

            # Urban Ladder (10 products)
            {"name": "Urban Ladder Study Table", "category": "desk", "brand": "Urban Ladder", "price": 12500, "style": "scandinavian", "width_cm": 120, "depth_cm": 60, "quality_score": 0.9, "link": "https://www.urbanladder.com/study-tables"},
            {"name": "Urban Ladder Engineered Wood Shelf", "category": "bookshelf", "brand": "Urban Ladder", "price": 8999, "style": "modern", "width_cm": 90, "depth_cm": 35, "quality_score": 0.85, "link": "https://www.urbanladder.com/bookshelves"},
            {"name": "Urban Ladder Crescent Study Chair", "category": "chair", "brand": "Urban Ladder", "price": 7999, "style": "contemporary", "width_cm": 60, "depth_cm": 60, "quality_score": 0.85, "link": "https://www.urbanladder.com/study-chairs"},
            {"name": "Urban Ladder Malabar Bookshelf", "category": "bookshelf", "brand": "Urban Ladder", "price": 14500, "style": "classic", "width_cm": 85, "depth_cm": 30, "quality_score": 0.9, "link": "https://www.urbanladder.com/bookshelves"},
            {"name": "Urban Ladder Oki Desk Organizer", "category": "organizer", "brand": "Urban Ladder", "price": 1200, "style": "minimalist", "width_cm": 30, "depth_cm": 20, "quality_score": 0.8, "link": "https://www.urbanladder.com/study-table-accessories"},
            {"name": "Urban Ladder Boston Study Table", "category": "desk", "brand": "Urban Ladder", "price": 18500, "style": "scandinavian", "width_cm": 115, "depth_cm": 60, "quality_score": 0.9, "link": "https://www.urbanladder.com/study-tables"},
            {"name": "Urban Ladder Aeris Desk Chair", "category": "chair", "brand": "Urban Ladder", "price": 11500, "style": "modern", "width_cm": 65, "depth_cm": 65, "quality_score": 0.88, "link": "https://www.urbanladder.com/study-chairs"},
            {"name": "Urban Ladder Carson Side Table", "category": "table", "brand": "Urban Ladder", "price": 4500, "style": "contemporary", "width_cm": 45, "depth_cm": 45, "quality_score": 0.82, "link": "https://www.urbanladder.com/side-tables"},
            {"name": "Urban Ladder Flora Table Lamp", "category": "lighting", "brand": "Urban Ladder", "price": 2500, "style": "modern", "width_cm": 22, "depth_cm": 22, "quality_score": 0.85, "link": "https://www.urbanladder.com/lamps"},
            {"name": "Urban Ladder Stanhope Desk", "category": "desk", "brand": "Urban Ladder", "price": 22000, "style": "scandinavian", "width_cm": 120, "depth_cm": 60, "quality_score": 0.92, "link": "https://www.urbanladder.com/study-tables"},

            # Godrej Interio (10 products)
            {"name": "Godrej Interio Caliber Desk", "category": "desk", "brand": "Godrej Interio", "price": 8500, "style": "contemporary", "width_cm": 110, "depth_cm": 55, "quality_score": 0.8, "link": "https://www.godrejinterio.com/furniture/study-table"},
            {"name": "Godrej Interio Motion Chair", "category": "chair", "brand": "Godrej Interio", "price": 12000, "style": "contemporary", "width_cm": 65, "depth_cm": 65, "quality_score": 0.85, "link": "https://www.godrejinterio.com/furniture/office-chair"},
            {"name": "Godrej Interio Companion Cabinet", "category": "cabinet", "brand": "Godrej Interio", "price": 9500, "style": "classic", "width_cm": 90, "depth_cm": 45, "quality_score": 0.8, "link": "https://www.godrejinterio.com/furniture/home-storage"},
            {"name": "Godrej Interio Unwind Chair", "category": "chair", "brand": "Godrej Interio", "price": 4500, "style": "modern", "width_cm": 55, "depth_cm": 55, "quality_score": 0.78, "link": "https://www.godrejinterio.com/furniture/chairs"},
            {"name": "Godrej Interio Script Desk", "category": "desk", "brand": "Godrej Interio", "price": 16500, "style": "contemporary", "width_cm": 120, "depth_cm": 60, "quality_score": 0.88, "link": "https://www.godrejinterio.com/furniture/study-table"},
            {"name": "Godrej Interio Premier Table", "category": "desk", "brand": "Godrej Interio", "price": 10500, "style": "classic", "width_cm": 115, "depth_cm": 60, "quality_score": 0.82, "link": "https://www.godrejinterio.com/furniture/study-table"},
            {"name": "Godrej Interio Slimline Wardrobe", "category": "wardrobe", "brand": "Godrej Interio", "price": 21000, "style": "classic", "width_cm": 90, "depth_cm": 50, "quality_score": 0.85, "link": "https://www.godrejinterio.com/furniture/wardrobe"},
            {"name": "Godrej Interio Versatile Shelf", "category": "bookshelf", "brand": "Godrej Interio", "price": 7500, "style": "modern", "width_cm": 80, "depth_cm": 30, "quality_score": 0.8, "link": "https://www.godrejinterio.com/furniture/home-storage"},
            {"name": "Godrej Interio Accent Lamp", "category": "lighting", "brand": "Godrej Interio", "price": 3000, "style": "contemporary", "width_cm": 25, "depth_cm": 25, "quality_score": 0.82, "link": "https://www.godrejinterio.com/"},
            {"name": "Godrej Interio Work Desk Combo", "category": "desk", "brand": "Godrej Interio", "price": 19999, "style": "contemporary", "width_cm": 120, "depth_cm": 60, "quality_score": 0.88, "link": "https://www.godrejinterio.com/furniture/study-table"},

            # Projectors & Projector Screens (5 products)
            {"name": "Epson EpiqVision Flex Projector", "category": "projector", "brand": "Epson", "price": 27389, "style": "modern", "width_cm": 30, "depth_cm": 30, "quality_score": 0.9, "image": "https://images.unsplash.com/photo-1535016120720-40c646be5580?auto=format&fit=crop&w=150&q=80", "link": "https://www.google.com/search?tbm=shop&q=Epson+EpiqVision+Flex+Projector"},
            {"name": "Vankyo Leisure Mini Projector", "category": "projector", "brand": "Vankyo", "price": 4979, "style": "modern", "width_cm": 20, "depth_cm": 15, "quality_score": 0.85, "image": "https://images.unsplash.com/photo-1535016120720-40c646be5580?auto=format&fit=crop&w=150&q=80", "link": "https://www.google.com/search?tbm=shop&q=Vankyo+Leisure+Mini+Projector"},
            {"name": "Vankyo Staytrue 120 Projector Screen", "category": "projector screen", "brand": "Vankyo", "price": 10789, "style": "modern", "width_cm": 260, "depth_cm": 10, "quality_score": 0.88, "image": "https://images.unsplash.com/photo-1583741893569-07301292b3bc?auto=format&fit=crop&w=150&q=80", "link": "https://www.google.com/search?tbm=shop&q=Vankyo+Staytrue+120+Projector+Screen"},
            {"name": "IKEA MITTzon Projector Frame", "category": "projector screen", "brand": "IKEA", "price": 26500, "style": "modern", "width_cm": 200, "depth_cm": 80, "quality_score": 0.86, "image": "https://images.unsplash.com/photo-1583741893569-07301292b3bc?auto=format&fit=crop&w=150&q=80", "link": "https://www.ikea.com/"},
            {"name": "Elite Screens Manual Projector Screen", "category": "projector screen", "brand": "Elite Screens", "price": 8999, "style": "classic", "width_cm": 240, "depth_cm": 12, "quality_score": 0.87, "image": "https://images.unsplash.com/photo-1583741893569-07301292b3bc?auto=format&fit=crop&w=150&q=80", "link": "https://www.google.com/search?tbm=shop&q=Elite+Screens+Manual+Projector+Screen"},
            
            # Additional Category Mappings & Brand Fallbacks (Beds, Green Soul)
            {"name": "IKEA Neiden Bed Frame", "category": "bed", "brand": "IKEA", "price": 8999, "style": "modern", "width_cm": 200, "depth_cm": 140, "quality_score": 0.8, "link": "https://www.ikea.com/in/en/p/neiden-bed-frame-pine-80395248/"},
            {"name": "IKEA Malm Bed Frame", "category": "bed", "brand": "IKEA", "price": 18999, "style": "modern", "width_cm": 210, "depth_cm": 160, "quality_score": 0.88, "link": "https://www.ikea.com/in/en/p/malm-bed-frame-high-w-2-storage-boxes-white-s79175979/"},
            {"name": "Urban Ladder Single Bed", "category": "bed", "brand": "Urban Ladder", "price": 22000, "style": "contemporary", "width_cm": 200, "depth_cm": 100, "quality_score": 0.86, "link": "https://www.urbanladder.com/beds"},
            {"name": "Godrej Interio Queen Bed", "category": "bed", "brand": "Godrej Interio", "price": 28000, "style": "classic", "width_cm": 210, "depth_cm": 150, "quality_score": 0.85, "link": "https://www.godrejinterio.com/furniture/beds"},
            {"name": "Green Soul Ergonomic Chair", "category": "chair", "brand": "Green Soul", "price": 8999, "style": "contemporary", "width_cm": 65, "depth_cm": 65, "quality_score": 0.9, "link": "https://www.greensoul.online/"},
            {"name": "Green Soul Study Desk", "category": "desk", "brand": "Green Soul", "price": 7500, "style": "modern", "width_cm": 120, "depth_cm": 60, "quality_score": 0.82, "link": "https://www.greensoul.online/"},
            
            # Air Conditioners (5 products for fallback)
            {"name": "Voltas 1.5 Ton 3 Star Split AC", "category": "air conditioner", "brand": "Voltas", "price": 32990, "style": "modern", "width_cm": 90, "depth_cm": 22, "quality_score": 0.85, "link": "https://www.google.com/search?tbm=shop&q=Voltas+1.5+Ton+3+Star+Split+AC"},
            {"name": "Daikin 1.5 Ton 5 Star Inverter AC", "category": "air conditioner", "brand": "Daikin", "price": 45490, "style": "modern", "width_cm": 90, "depth_cm": 22, "quality_score": 0.92, "link": "https://www.google.com/search?tbm=shop&q=Daikin+1.5+Ton+5+Star+Inverter+AC"},
            {"name": "LG 1.5 Ton 5 Star Dual Inverter AC", "category": "air conditioner", "brand": "LG", "price": 46500, "style": "modern", "width_cm": 90, "depth_cm": 22, "quality_score": 0.9, "link": "https://www.google.com/search?tbm=shop&q=LG+1.5+Ton+5+Star+Dual+Inverter+AC"},
            {"name": "Samsung 1 Ton 3 Star Inverter AC", "category": "air conditioner", "brand": "Samsung", "price": 28990, "style": "modern", "width_cm": 80, "depth_cm": 20, "quality_score": 0.82, "link": "https://www.google.com/search?tbm=shop&q=Samsung+1+Ton+3+Star+Inverter+AC"},
            {"name": "Lloyd 1.5 Ton 3 Star Split AC", "category": "air conditioner", "brand": "Lloyd", "price": 32990, "style": "modern", "width_cm": 90, "depth_cm": 22, "quality_score": 0.83, "link": "https://www.google.com/search?tbm=shop&q=Lloyd+1.5+Ton+3+Star+Split+AC"},
            
            # Ceiling Fans (3 products for fallback)
            {"name": "Ceiling Fan Hugger 52 in. LED Indoor Brushed Nickel Ceiling Fan", "category": "ceiling fan", "brand": "Generic", "price": 4977.51, "style": "modern", "width_cm": 130, "depth_cm": 130, "quality_score": 0.88, "link": "https://www.google.com/search?tbm=shop&q=Ceiling+Fan+Hugger+52+in.+LED+Indoor+Brushed+Nickel+Ceiling+Fan"},
            {"name": "Hugger 52 in. LED Indoor Black Ceiling Fan", "category": "ceiling fan", "brand": "Generic", "price": 4977.51, "style": "modern", "width_cm": 130, "depth_cm": 130, "quality_score": 0.87, "link": "https://www.google.com/search?tbm=shop&q=Hugger+52+in.+LED+Indoor+Black+Ceiling+Fan"},
            {"name": "Visual Comfort Fan Ikon 44\" Ceiling Fan", "category": "ceiling fan", "brand": "Generic", "price": 26493.60, "style": "modern", "width_cm": 110, "depth_cm": 110, "quality_score": 0.9, "link": "https://www.google.com/search?tbm=shop&q=Visual+Comfort+Fan+Ikon+44%22+Ceiling+Fan"}
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
        max_budget = float(budget_range.get("maximum") or 50000.0)
        pref_style = user_preferences.get("preferred_style") or "modern"

        # Robust category matching helper function to handle variations like "queen-size bed" vs "bed"
        def is_category_match(item_cat: str, req_cat: str) -> bool:
            c1 = str(item_cat).lower()
            c2 = str(req_cat).lower()
            words = ["bed", "desk", "chair", "table", "bookshelf", "shelf", "fan", "ac", "conditioner", "projector", "screen", "lamp", "light"]
            for w in words:
                if w in c1 and w in c2:
                    return True
            return c1 == c2 or c2 in c1 or c1 in c2

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
        candidates = self.filter_dimensions(candidates, max_w=150, max_d=80)
        
        # Calculate budget per item to fit the total budget limit
        num_items = len(req_products) if req_products else 1
        budget_per_item = max_budget / num_items if num_items > 1 else max_budget
        
        # Try to filter by budget_per_item first, falling back to max_budget if empty
        candidates_under_item_limit = [c for c in candidates if float(c.get("price", 0)) <= budget_per_item]
        
        # Check if we still have options for each required category
        has_all_cats = True
        for cat in req_products:
            normalized_key = cat.lower().replace(" ", "_")
            cat_found = False
            for c in candidates_under_item_limit:
                if (is_category_match(c.get("category", ""), cat) or 
                    is_category_match(c.get("name", ""), cat) or 
                    is_category_match(c.get("title", ""), cat)):
                    cat_found = True
                    break
            if not cat_found:
                has_all_cats = False
                break
                
        if has_all_cats:
            candidates = candidates_under_item_limit
        else:
            # Fall back to individual max_budget limit
            candidates = [c for c in candidates if float(c.get("price", 0)) <= max_budget]
        
        # Exclude brand filter
        candidates = [c for c in candidates if c.get("brand") not in excluded_brands]

        # 3. Score candidates using the conformed 5-factor formulas
        scored = []
        for c in candidates:
            scores = self.score_products(
                product=c,
                pref_style=pref_style,
                preferred_brands=preferred_brands,
                required_w=120,
                required_d=60,
                budget_limit=max_budget,
                spatial_output=spatial_output
            )
            item = dict(c)
            item.update(scores)
            # Add conformed schema score aliases matching Section 1
            item["fit_score"] = scores.get("fit_score", 0.8)
            item["design_score"] = scores.get("design_score", 0.8)
            item["rank_score"] = scores.get("final_score", 0.8)
            scored.append(item)

        # 4. Rank products
        ranked = self.rank_products(scored)
        
        # Keep top 10 per category first to prevent crowding out
        category_grouped = {}
        for cat in req_products:
            normalized_key = cat.lower().replace(" ", "_")
            cat_list = []
            for item in ranked:
                if (is_category_match(item.get("category", ""), cat) or 
                    is_category_match(item.get("name", ""), cat) or 
                    is_category_match(item.get("title", ""), cat)):
                    cat_list.append(item)
            category_grouped[normalized_key] = cat_list[:5] # Keep top 5 per category

        # Union top 3 of each category for the main recommended list
        curated_ranked = []
        for cat_list in category_grouped.values():
            curated_ranked.extend(cat_list[:3])
        curated_ranked = self.rank_products(curated_ranked)

        # 5. Build shopping list & explanations
        shopping_list = self.build_shopping_list(curated_ranked)
        explanations = self.generate_explanations(curated_ranked)

        output = {
            "recommended_products": curated_ranked,
            "shopping_list": shopping_list,
            "explanations": explanations
        }

        # Populate category-specific outputs
        for k, v in category_grouped.items():
            output[k] = v

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
        api_key = os.environ.get("SERP_API_KEY") or os.environ.get("SERPAPI_API_KEY") or getattr(config, "SERP_API", None)
        if not api_key:
            # Try UserSecretsClient if on Kaggle
            try:
                from kaggle_secrets import UserSecretsClient
                api_key = UserSecretsClient().get_secret("SERP_API_KEY") or UserSecretsClient().get_secret("SERPAPI_API_KEY")
            except BaseException:
                pass

        if not api_key:
            logger.warning("SerpAPI API key not found. Falling back to local catalog search.")
            return self._search_mock_catalog(categories)

        import urllib.request
        import urllib.parse
        import urllib.error
        
        results = []
        
        # Helper function to parse a single SerpAPI shopping item into conformed schema
        def parse_item(item, cat, detected_brand):
            title = item.get("title", "")
            price_str = item.get("price", "0")
            # Extract price digits
            price_cleaned = re.sub(r'[^\d.]', '', price_str)
            try:
                price = float(price_cleaned) if price_cleaned else 0.0
            except Exception:
                price = 0.0

            # Dynamic currency conversion to INR
            converted_price = price
            is_usd = False
            is_eur = False
            is_gbp = False
            is_cad = False
            
            price_str_lower = price_str.lower()
            currency_field = item.get("currency", "").upper()
            
            if "$" in price_str or "usd" in price_str_lower or currency_field == "USD":
                is_usd = True
            elif "€" in price_str or "eur" in price_str_lower or currency_field == "EUR":
                is_eur = True
            elif "£" in price_str or "gbp" in price_str_lower or currency_field == "GBP":
                is_gbp = True
            elif "c$" in price_str_lower or "cad" in price_str_lower or currency_field == "CAD":
                is_cad = True
                
            if is_usd:
                converted_price = round(price * 83.0, 2)
            elif is_eur:
                converted_price = round(price * 90.0, 2)
            elif is_gbp:
                converted_price = round(price * 105.0, 2)
            elif is_cad:
                converted_price = round(price * 61.0, 2)

            price = converted_price
            if price <= 0:
                price = 4500.0
                
            rating = float(item.get("rating", 4.0))
            quality_score = rating / 5.0 if rating > 0 else 0.8
            reviews_count = int(item.get("reviews", 50))
            image_url = item.get("thumbnail", "")
            
            # Parse dimensions from title
            w_cm, d_cm, h_cm = 120.0, 60.0, 75.0
            match = re.search(r'(\d+)\s*(?:x|by)\s*(\d+)', title.lower())
            if match:
                try:
                    w_cm = float(match.group(1))
                    d_cm = float(match.group(2))
                except Exception:
                    pass
                    
            # Build conformed product schema defined in Section 1
            product_id = item.get("product_id") or f"serp_{hash(title)%1000}"
            orig_link = item.get("product_link") or item.get("link") or item.get("serpapi_product_api") or ""
            if not orig_link or not (orig_link.startswith("http://") or orig_link.startswith("https://")):
                query_q = f"{detected_brand} {title}".strip()
                orig_link = f"https://www.google.com/search?tbm=shop&q={urllib.parse.quote(query_q)}"
            
            return {
                # Dynamic conformed Section 1 schema fields
                "product_id": product_id,
                "title": title,
                "brand": detected_brand,
                "price": price,
                "currency": "INR",
                "rating": rating,
                "reviews": reviews_count,
                "dimensions": {
                    "width_cm": w_cm,
                    "depth_cm": d_cm,
                    "height_cm": h_cm
                },
                "thumbnail_url": image_url,
                "high_resolution_image": image_url,
                "product_url": orig_link,
                "seller": detected_brand,
                "source": "serpapi",
                "marketplace": detected_brand,
                "availability": True,
                "shipping_cost": 599,
                "installation_cost": 0,
                
                # Backwards-compatibility legacy properties
                "name": title,
                "category": cat,
                "style": "modern",
                "width_cm": w_cm,
                "depth_cm": d_cm,
                "quality_score": round(quality_score, 2),
                "image": image_url,
                "link": orig_link
            }

        # Process each category independently to ensure we never truncate entire categories!
        for cat in categories or ["desk"]:
            cat_results = []
            
            # Step A: Try searching with preferred brands first
            if preferred_brands:
                for brand in preferred_brands[:2]: # Limit to top 2 brands to avoid query limit hits
                    query = f"{brand} {cat}".strip()
                    try:
                        logger.info("Executing branded SerpAPI query: '%s'", query)
                        query_encoded = urllib.parse.quote(query)
                        url = f"https://serpapi.com/search.json?engine=google_shopping&q={query_encoded}&api_key={api_key}"
                        
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=12) as response:
                            data = json.loads(response.read().decode('utf-8'))
                            shopping_results = data.get("shopping_results", [])
                            
                            for item in shopping_results:
                                parsed = parse_item(item, cat, brand)
                                if parsed:
                                    cat_results.append(parsed)
                    except Exception as e:
                        logger.warning("Branded query failed for '%s': %s", query, str(e))
                        
            # Step B: Generic search fallback
            # If branded search returned 0 items, or if no preferred brands are set:
            # We run a generic category search to ensure we ALWAYS return products of ANY brand!
            if not cat_results:
                query = cat
                try:
                    logger.info("Executing generic fallback SerpAPI query: '%s'", query)
                    query_encoded = urllib.parse.quote(query)
                    url = f"https://serpapi.com/search.json?engine=google_shopping&q={query_encoded}&api_key={api_key}"
                    
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=12) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        shopping_results = data.get("shopping_results", [])
                        
                        for item in shopping_results:
                            source = item.get("source", "")
                            detected_brand = item.get("brand") or source or "Generic"
                            parsed = parse_item(item, cat, detected_brand)
                            if parsed:
                                cat_results.append(parsed)
                except Exception as e:
                    logger.warning("Generic query failed for '%s': %s", query, str(e))
                    
            # Step C: Category fallback check if live results are still empty
            if not cat_results:
                logger.info("SerpAPI failed completely for category '%s'. Fetching fallbacks from local mock catalog...", cat)
                mock_items = self._search_mock_catalog([cat])
                cat_results.extend(mock_items)
                
            results.extend(cat_results)
            
        return results

    def _search_mock_catalog(self, categories: List[str]) -> List[Dict[str, Any]]:
        """Fallback search logic using local mock catalog."""
        if not categories:
            matches = [c for c in self.mock_catalog if c.get("category") == "desk"]
        else:
            matches = []
            for cat in categories:
                normalized_cat = cat.lower()
                words = ["bed", "desk", "chair", "table", "bookshelf", "shelf", "fan", "ac", "conditioner", "projector", "screen", "lamp", "light"]
                match_words = [w for w in words if w in normalized_cat]
                
                cat_matches = []
                for c in self.mock_catalog:
                    item_cat = c.get("category", "").lower()
                    item_name = c.get("name", "").lower()
                    matched = False
                    if match_words:
                        for w in match_words:
                            if w in item_cat or w in item_name:
                                matched = True
                                break
                    if not matched and (normalized_cat in item_cat or normalized_cat in item_name or item_cat in normalized_cat):
                        matched = True
                    if matched:
                        cat_matches.append(c)
                matches.extend(cat_matches)
        
        if not matches:
            matches = list(self.mock_catalog)

        # Enrich with reviews, rating, image, link conformed to Section 1 schema
        enriched = []
        for idx, item in enumerate(matches):
            rating = item.get("rating", round(item.get("quality_score", 0.8) * 5.0, 1))
            reviews = item.get("reviews", 50)
            
            # Resolve a beautiful, high-fidelity Unsplash product image based on the item's category
            cat_lower = item.get("category", "").lower()
            if "desk" in cat_lower or "table" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?auto=format&fit=crop&w=150&q=80"
            elif "chair" in cat_lower or "stool" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1505797149-43b0069ec26b?auto=format&fit=crop&w=150&q=80"
            elif "bookshelf" in cat_lower or "shelf" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1594980596870-8cb524ebf6bc?auto=format&fit=crop&w=150&q=80"
            elif "bed" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=150&q=80"
            elif "ac" in cat_lower or "conditioner" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1585338107529-13afc5f02586?auto=format&fit=crop&w=150&q=80"
            elif "projector" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1535016120720-40c646be5580?auto=format&fit=crop&w=150&q=80"
            elif "lighting" in cat_lower or "lamp" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?auto=format&fit=crop&w=150&q=80"
            elif "cabinet" in cat_lower or "wardrobe" in cat_lower:
                img_url = "https://images.unsplash.com/photo-1595428774223-ef52624120d2?auto=format&fit=crop&w=150&q=80"
            else:
                img_url = "https://images.unsplash.com/photo-1513151233558-d860c5398176?auto=format&fit=crop&w=150&q=80"
                
            thumbnail_url = item.get("image") or img_url
            high_res = thumbnail_url.replace("w=150", "w=800") if "unsplash.com" in thumbnail_url else thumbnail_url
            
            p_url = item.get("link")
            if not p_url:
                import urllib.parse
                search_q = urllib.parse.quote(f"{item.get('brand')} {item.get('name')}")
                p_url = f"https://www.google.com/search?tbm=shop&q={search_q}"
                
            w_cm = float(item.get("width_cm", 120))
            d_cm = float(item.get("depth_cm", 60))
            h_cm = float(item.get("height_cm", 75))
            
            conformed_product = {
                # Dynamic conformed Section 1 schema fields
                "product_id": f"mock_{idx + 1:03d}",
                "title": item.get("name"),
                "brand": item.get("brand"),
                "price": float(item.get("price", 0)),
                "currency": "INR",
                "rating": float(rating),
                "reviews": int(reviews),
                "dimensions": {
                    "width_cm": w_cm,
                    "depth_cm": d_cm,
                    "height_cm": h_cm
                },
                "thumbnail_url": thumbnail_url,
                "high_resolution_image": high_res,
                "product_url": p_url,
                "seller": item.get("brand"),
                "source": "mock_catalog",
                "marketplace": item.get("brand"),
                "availability": True,
                "shipping_cost": 0,
                "installation_cost": 0,
                
                # Backwards-compatibility legacy properties
                "name": item.get("name"),
                "category": item.get("category"),
                "style": item.get("style", "modern"),
                "width_cm": w_cm,
                "depth_cm": d_cm,
                "quality_score": float(item.get("quality_score", 0.8)),
                "image": thumbnail_url,
                "link": p_url
            }
            enriched.append(conformed_product)
            
        return enriched

    def filter_dimensions(self, products: List[Dict[str, Any]], max_w: float, max_d: float) -> List[Dict[str, Any]]:
        """Filters out items that exceed the available room clearance boundaries."""
        filtered = []
        for p in products:
            w = float(p.get("width_cm", 0))
            d = float(p.get("depth_cm", 0))
            cat_lower = p.get("category", "").lower()
            
            # Set category-specific dimension boundaries to prevent false filtering of large items (Beds/Screens)
            limit_w = max_w
            limit_d = max_d
            
            if "bed" in cat_lower:
                limit_w = 250.0
                limit_d = 220.0
            elif "screen" in cat_lower or "tv" in cat_lower:
                limit_w = 300.0
                limit_d = 100.0
            elif "projector" in cat_lower or "ac" in cat_lower or "conditioner" in cat_lower or "lighting" in cat_lower or "fan" in cat_lower:
                limit_w = 200.0
                limit_d = 200.0
                
            if w <= limit_w and d <= limit_d:
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
        budget_limit: float,
        spatial_output: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Calculates conformed weighted score for the product candidate.

        Formula:
          Final Score = 0.40 * fit_score + 0.25 * rating_score + 0.15 * design_score + 0.10 * brand_score + 0.10 * popularity_score
        """
        # 1. fit_score: how well dimensions fit the empty regions/clearance boundaries (40%)
        empty_regions = []
        if spatial_output:
            empty_regions = spatial_output.get("empty_regions") or []
        prod_w = float(product.get("width_cm") or required_w)
        prod_d = float(product.get("depth_cm") or required_d)
        
        fit_score = 0.3
        if empty_regions:
            for region in empty_regions:
                rw = float(region.get("width_cm") or 0.0)
                rd = float(region.get("depth_cm") or 0.0)
                if prod_w <= rw and prod_d <= rd:
                    fit_score = 1.0
                    break
        else:
            fit_score = 1.0 if (prod_w <= required_w and prod_d <= required_d) else 0.5

        # 2. rating_score (25%)
        rating = float(product.get("rating") or product.get("quality_score", 0.8) * 5.0 or 4.0)
        rating_score = min(1.0, max(0.0, rating / 5.0))

        # 3. design_score: style compatibility (15%)
        style_match = product.get("style", "").lower() == pref_style.lower()
        design_score = 1.0 if style_match else 0.6

        # 4. brand_score: brand preference (10%)
        brand_match = product.get("brand", "").lower() in [b.lower() for b in preferred_brands]
        brand_score = 1.0 if brand_match else 0.5

        # 5. popularity_score: based on reviews count or fallback (10%)
        reviews_count = int(product.get("reviews", 50))
        popularity_score = min(1.0, reviews_count / 100.0)

        # Final weighted score
        final_score = (
            0.40 * fit_score +
            0.25 * rating_score +
            0.15 * design_score +
            0.10 * brand_score +
            0.10 * popularity_score
        )
        
        return {
            "fit_score": round(fit_score, 2),
            "rating_score": round(rating_score, 2),
            "design_score": round(design_score, 2),
            "brand_score": round(brand_score, 2),
            "popularity_score": round(popularity_score, 2),
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
