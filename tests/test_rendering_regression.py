import sys
import os
import unittest
from PIL import Image, ImageChops

# Ensure the root path is in the sys path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.reconstruction_agent import make_background_transparent

class TestRenderingRegression(unittest.TestCase):
    def test_make_background_transparent_fallback(self):
        # Create a completely white image to simulate a product thumbnail
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        
        # Apply transparency conversion
        result = make_background_transparent(img)
        
        # Verify the output is not None and has the same size
        self.assertIsNotNone(result)
        self.assertEqual(result.size, (100, 100))
        
        # Check that it did not return a completely transparent image
        bbox = result.getbbox()
        self.assertIsNotNone(bbox, "Resulting image should not be completely transparent.")

    def test_pixel_level_rendering_difference(self):
        # Create a mock base room image (500x500 blue room canvas)
        room_img = Image.new("RGBA", (500, 500), (41, 128, 185, 255))
        
        # Create a mock product image (50x50 solid red square)
        product_img = Image.new("RGBA", (50, 50), (231, 76, 60, 255))
        
        # Make a copy of the room image and paste the product
        reconstructed = room_img.copy()
        reconstructed.paste(product_img, (100, 100), product_img)
        
        # Compute image differences
        diff = ImageChops.difference(room_img.convert("RGB"), reconstructed.convert("RGB"))
        bbox = diff.getbbox()
        
        # The images must differ at the pasted region
        self.assertIsNotNone(bbox, "Pasted product must result in pixel differences.")
        
        # Check that the difference bounding box matches the pasted location approximately
        self.assertTrue(bbox[0] <= 100 <= bbox[2])
        self.assertTrue(bbox[1] <= 100 <= bbox[3])

if __name__ == "__main__":
    unittest.main()
