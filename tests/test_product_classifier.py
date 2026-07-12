import sys
import os
import unittest

# Ensure the root path is in the sys path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.product_classifier import ProductClassifier

class TestProductClassifier(unittest.TestCase):
    def test_step_13_items(self):
        test_cases = [
            (
                {"name": "IKEA Ceiling Fan", "category": "Ceiling Fan"},
                "ceiling_fan", "ceiling"
            ),
            (
                {"name": "Samsung Smart TV", "category": "Television"},
                "television", "wall_or_surface"
            ),
            (
                {"name": "IKEA BILLY Bookcase", "category": "Bookcase"},
                "bookcase", "floor_wall_adjacent"
            ),
            (
                {"name": "Study Table", "category": "Furniture"},
                "study_table", "floor"
            ),
            (
                {"name": "Ergonomic Office Chair", "category": "Chair"},
                "office_chair", "floor"
            ),
            (
                {"name": "Coffee Maker", "category": "Kitchen Appliance"},
                "coffee_maker", "support_surface"
            ),
            (
                {"name": "Laptop Stand", "category": "Accessories"},
                "laptop_holder", "support_surface"
            ),
            (
                {"name": "Air Cooler", "category": "Appliance"},
                "air_cooler", "floor"
            )
        ]
        
        for product, expected_cat, expected_dom in test_cases:
            cat, dom = ProductClassifier.classify(product)
            self.assertEqual(cat, expected_cat, f"Expected category {expected_cat} for {product}, got {cat}")
            self.assertEqual(dom, expected_dom, f"Expected domain {expected_dom} for {product}, got {dom}")

    def test_none_product(self):
        cat, dom = ProductClassifier.classify(None)
        self.assertEqual(cat, "unknown")
        self.assertEqual(dom, "unknown")

    def test_empty_dict(self):
        cat, dom = ProductClassifier.classify({})
        self.assertEqual(cat, "unknown")
        self.assertEqual(dom, "unknown")

    def test_title_instead_of_name(self):
        cat, dom = ProductClassifier.classify({"title": "Minimalist Study Table"})
        self.assertEqual(cat, "study_table")
        self.assertEqual(dom, "floor")

    def test_product_name_field(self):
        cat, dom = ProductClassifier.classify({"product_name": "Portable Air Cooler"})
        self.assertEqual(cat, "air_cooler")
        self.assertEqual(dom, "floor")

    def test_product_category_field(self):
        cat, dom = ProductClassifier.classify({"product_category": "Office Chair"})
        self.assertEqual(cat, "office_chair")
        self.assertEqual(dom, "floor")

    def test_mixed_capitalization(self):
        cat, dom = ProductClassifier.classify({"name": "SMART TV"})
        self.assertEqual(cat, "television")
        self.assertEqual(dom, "wall_or_surface")

    def test_projector_screen(self):
        cat, dom = ProductClassifier.classify({"name": "120 inch Projector Screen"})
        self.assertEqual(cat, "projector_screen")
        self.assertEqual(dom, "wall")

    def test_tv_unit(self):
        cat, dom = ProductClassifier.classify({"name": "Modern TV Unit"})
        self.assertEqual(cat, "tv_unit")
        self.assertEqual(dom, "floor")

    def test_unknown_product(self):
        cat, dom = ProductClassifier.classify({"name": "Unknown Decorative Object"})
        self.assertEqual(dom, "unknown")

    def test_original_metadata_preservation(self):
        orig = {"name": "Test Table", "price": 4500, "link": "http://example.com"}
        normalized = ProductClassifier.normalize_product(orig)
        self.assertEqual(normalized["name"], "Test Table")
        self.assertEqual(normalized["price"], 4500)
        self.assertEqual(normalized["link"], "http://example.com")
        self.assertEqual(normalized["normalized_name"], "test table")

if __name__ == "__main__":
    unittest.main()
