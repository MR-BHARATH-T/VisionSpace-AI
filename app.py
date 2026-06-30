import os
import sys
import pprint
from PIL import Image

# Ensure the root path is in the sys path
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.vision_agent import VisionAgent
from agents.spatial_agent import SpatialAgent
from agents.design_agent import DesignAgent

def main():
    print("==================================================")
    # 1. Initialize Agents
    print("Initializing Multi-Agent Pipeline...")
    vision_agent = VisionAgent()
    spatial_agent = SpatialAgent()
    design_agent = DesignAgent()
    
    # Create a dummy image for pipeline demonstration
    print("Creating dummy room image...")
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    temp_img_path = "temp_room_view.jpg"
    img.save(temp_img_path)
    
    try:
        # 2. Run Vision Agent (What exists in the room?)
        print("\n--- Step 1: Running Vision Agent ---")
        vision_output = vision_agent.analyze_room(
            images=[temp_img_path],
            room_type="bedroom",
            style_preference="scandinavian"
        )
        session_id = vision_output["session_id"]
        session_path = vision_output["metadata"]["session_path"]
        print(f"Vision Agent finished. Session ID: {session_id}")
        print(f"Output files stored in: {session_path}")
        
        # 3. Run Spatial Agent (What can fit in the room?)
        print("\n--- Step 2: Running Spatial Agent ---")
        spatial_output = spatial_agent.analyze_layout(
            vision_output=vision_output,
            proposed_furniture=None,
            session_path=session_path
        )
        print("Spatial Agent finished layout analysis successfully.")
        
        # 4. Run Design Agent (What SHOULD this room become?)
        print("\n--- Step 3: Running Design Agent ---")
        design_output = design_agent.generate_recommendations(
            vision_output=vision_output,
            spatial_output=spatial_output,
            budget=50000,
            preferred_style="scandinavian",
            preferred_colors=["white", "oak", "light_grey"],
            preferred_brands=["IKEA", "Urban Ladder"],
            constraints={
                "retain_existing_bed": True,
                "keep_window_access": True,
                "wheelchair_accessible": False,
                "work_from_home": True
            },
            session_path=session_path
        )
        
        print("\n==================================================")
        print("PIPELINE RUN COMPLETED SUCCESSFULLY!")
        print("==================================================")
        print("\nDesign Theme generated:")
        print(design_output.get("design_theme"))
        print("\nColor Palette:")
        print(design_output.get("color_palette"))
        print("\nRecommended Furniture count:")
        print(len(design_output.get("recommended_furniture", [])))
        print("\nLayout Plan positions:")
        pprint.pprint(design_output.get("layout_plan"))
        print("\nDesign Explanations:")
        for explanation in design_output.get("design_explanations", []):
            print(f"- {explanation}")
            
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    main()