import os
import sys
import shutil
import pathlib
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
    print("Initializing Multi-Agent Pipeline...")
    vision_agent = VisionAgent()
    spatial_agent = SpatialAgent()
    design_agent = DesignAgent()
    
    # Check for default assets
    fallback_paths = [
        "assets/front-view.png", 
        "assets/left-view.png", 
        "assets/right-view.png"
    ]
    images = []
    
    for path in fallback_paths:
        p = pathlib.Path(path)
        if p.exists():
            img = Image.open(p)
            images.append(img)
            print(f"  Loaded local asset: {p.name} ({img.width}x{img.height} px)")
        else:
            print(f"  Fallback file not found at: {path}")

    # Fallback to creating a dummy image if assets folder is completely missing
    if not images:
        print("  Creating fallback dummy room image...")
        img = Image.new("RGB", (400, 400), color=(240, 240, 240))
        images.append(img)
    
    try:
        # 1. Run Vision Agent (What exists?)
        print("\n🏠 Step 1: Running Vision Agent...")
        vision_output = vision_agent.analyze_room(
            images=images,
            room_type="living_room",
            style_preference="modern",
            progress_callback=lambda step: print(f"  [Step] {step}")
        )
        session_id = vision_output["session_id"]
        session_path = vision_output["metadata"]["session_path"]
        print(f"Vision Agent finished. Session ID: {session_id}")

        # Explicitly unload vision model to free VRAM before loading text models
        print("Unloading vision model to free VRAM...")
        vision_agent.model_manager.unload_model("vision")
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        # 2. Run Spatial Agent (What can fit?)
        print("\n🧠 Step 2: Running Spatial Agent...")
        spatial_output = spatial_agent.analyze_layout(
            room_data=vision_output,
            proposed_furniture=None,
            session_path=session_path
        )
        print("Spatial Agent finished layout analysis successfully.")
        
        # 3. Run Design Agent (What SHOULD this room become?)
        print("\n🌌 Step 3: Running Design Agent...")
        design_output = design_agent.generate_recommendations(
            vision_output=vision_output,
            spatial_output=spatial_output,
            budget=50000,
            preferred_style="modern",
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
        print("Design Agent recommendations completed successfully.")
        
        print("\n✅ Multi-Agent Pipeline Analysis Complete!")
        print(f"Session ID: {session_id}")
        print(f"All spatial and design outputs are stored in: {session_path}")
        
        print("\n--- Output Files Exported ---")
        print(f"1. Unified Scene Graph: {session_path}/scene_graph.json")
        print(f"2. Occupancy Grid matrix: {session_path}/occupancy_grid.json")
        print(f"3. Merged Scene: {session_path}/merged_output.json")
        print(f"4. Spatial Layout Analysis: {session_path}/spatial_analysis.json")
        print(f"5. Design Recommendation: {session_path}/design_analysis.json")
        
        # Zip outputs for ease of downloading
        print("\n📦 Zipping output files...")
        outputs_dir = pathlib.Path("outputs")
        zip_output_path = pathlib.Path("../outputs")  # outside workspace directory /kaggle/working/
        
        if outputs_dir.exists():
            shutil.make_archive(str(zip_output_path), 'zip', str(outputs_dir))
            print("Zipping complete! Created outputs.zip successfully.")
        else:
            print("Error: The outputs directory does not exist.")
            
    except Exception as e:
        print(f"\n❌ Pipeline Execution Failed: {str(e)}")
        raise e
    finally:
        try:
            print("\n🧹 Reclaiming all VRAM resources...")
            from models.model_manager import ModelManager
            ModelManager().unload_model("vision")
            ModelManager().unload_model("reasoning")
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

if __name__ == "__main__":
    main()