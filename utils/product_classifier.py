import re
from typing import Any, Dict, Tuple

class ProductClassifier:
    @staticmethod
    def normalize_product(product: Dict[str, Any]) -> Dict[str, Any]:
        """Safely normalizes product dictionary keys and creates a copy with normalized metadata.
        Does not mutate or overwrite original values, but appends normalized fields.
        """
        if not isinstance(product, dict):
            product = {}
        
        prod_copy = dict(product)
        
        prod_name = str(
            prod_copy.get("name")
            or prod_copy.get("title")
            or prod_copy.get("product_name")
            or ""
        ).strip()
        
        prod_cat = str(
            prod_copy.get("category")
            or prod_copy.get("product_category")
            or prod_copy.get("item_type")
            or prod_copy.get("type")
            or ""
        ).strip()
        
        prod_brand = str(
            prod_copy.get("brand")
            or ""
        ).strip()
        
        # Add conformed normalized metadata for internal/downstream use
        prod_copy["normalized_name"] = prod_name.lower()
        prod_copy["normalized_category"] = prod_cat.lower()
        prod_copy["normalized_brand"] = prod_brand.lower()
        
        return prod_copy

    @staticmethod
    def classify(product: Dict[str, Any]) -> Tuple[str, str]:
        """Classifies product into canonical_category and placement_domain."""
        if not product or not isinstance(product, dict):
            return "unknown", "unknown"
            
        prod = ProductClassifier.normalize_product(product)
        name = prod["normalized_name"]
        cat = prod["normalized_category"]
        
        text = f"{name} {cat}".strip()
        
        # 1. Ceiling Fan (Must check phrase/specific keywords first)
        if "ceiling fan" in text or text == "fan" or ("ceiling" in text and "fan" in text):
            return "ceiling_fan", "ceiling"
            
        # 2. Projector Screen (Check before generic projector)
        if "projector screen" in text:
            return "projector_screen", "wall"
            
        # 3. Projector
        if "projector" in text:
            return "projector", "ceiling_or_surface"
            
        # 4. TV Unit
        if any(k in text for k in ["tv unit", "tv stand", "media console", "entertainment unit", "entertainment center"]):
            return "tv_unit", "floor"
            
        # 5. Television (Smart TV, Led TV etc. Check after TV Unit)
        if any(k in text for k in ["television", "smart tv", "led tv", "oled tv", "qled tv"]):
            return "television", "wall_or_surface"
        if re.search(r'\btv\b', text):
            return "television", "wall_or_surface"
            
        # 6. Bookcase / Bookshelf
        if any(k in text for k in ["bookcase", "bookshelf", "book shelf"]):
            return "bookcase", "floor_wall_adjacent"
            
        # 7. Study Table
        if any(k in text for k in ["study table", "computer table", "office desk", "writing desk", "desk"]):
            return "study_table", "floor"
            
        # 8. Office Chair
        if any(k in text for k in ["office chair", "study chair", "ergonomic chair", "computer chair"]):
            return "office_chair", "floor"
            
        # 9. Coffee Maker
        if any(k in text for k in ["coffee maker", "coffee machine"]):
            return "coffee_maker", "support_surface"
            
        # 10. Laptop Holder
        if any(k in text for k in ["laptop holder", "laptop stand"]):
            return "laptop_holder", "support_surface"
            
        # 11. Air Cooler
        if any(k in text for k in ["air cooler", "portable cooler", "cooler"]):
            return "air_cooler", "floor"
            
        # 12. Wall Art
        if any(k in text for k in ["wall art", "artwork", "painting"]):
            return "wall_art", "wall"
            
        # 13. Air Conditioner
        if any(k in text for k in ["air conditioner", "split ac", "wall ac", "conditioner"]):
            return "air_conditioner", "wall"
            
        # Fallback
        orig_cat = product.get("category") or product.get("product_category") or product.get("item_type") or product.get("type") or "unknown"
        return str(orig_cat).strip(), "unknown"
