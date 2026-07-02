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

from workflow.langgraph_workflow import ProductionPipelineV2

def main():
    print("==================================================")
    print("Initializing Multi-Agent Pipeline (v2)...")
    pipeline = ProductionPipelineV2()
    
    # Check for uploaded images first in the uploads/ folder
    uploads_dir = pathlib.Path("uploads")
    images = []
    
    if uploads_dir.exists() and uploads_dir.is_dir():
        for path in uploads_dir.iterdir():
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                try:
                    img = Image.open(path)
                    images.append(img)
                    print(f"  Loaded uploaded image: {path.name} ({img.width}x{img.height} px)")
                except Exception as e:
                    print(f"  Error loading upload {path.name}: {e}")

    # Fallback to default assets if uploads/ is empty
    if not images:
        fallback_paths = [
            "assets/front-view.png", 
            "assets/left-view.png", 
            "assets/right-view.png"
        ]
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
        # Phase 0: User Preference Collection
        print("\n📥 Step 0: Collecting User Preferences...")
        user_prefs = {
            "required_products": ["study table", "office chair"],
            "preferred_brands": ["IKEA", "Godrej Interio"],
            "excluded_brands": ["Local"],
            "budget_range": {
                "minimum": 10000,
                "maximum": 50000
            },
            "preferred_style": "modern",
            "color_preferences": ["white", "wood"],
            "priority": ["workspace", "budget"]
        }
        
        # Run Phase 1 - Phase 7 Production Pipeline v2
        print("\n🚀 Executing Production Pipeline v2 (Phase 1 to Phase 7)...")
        pipeline_output = pipeline.run_pipeline(
            images=images,
            user_preferences=user_prefs,
            room_type="bedroom",
            progress_callback=lambda step: print(f"  [Step] {step}")
        )
        
        session_id = pipeline_output["session_id"]
        session_path = pipeline_output["session_path"]
        
        print("\n✅ Multi-Agent Pipeline Analysis Complete!")
        print(f"Session ID: {session_id}")
        print(f"All outputs are stored in: {session_path}")
        
        # Display Top 10 Recommended Products from Preferred Brands
        print("\n🛍️ Top 10 Best Recommended Products from Preferred Brands:")
        rec_products = pipeline_output.get("commerce", {}).get("recommended_products", [])
        for idx, item in enumerate(rec_products[:10]):
            print(f"  {idx+1:02d}. {item.get('brand')} - {item.get('name')} (Price: Rs. {item.get('price')}) | Style: {item.get('style')} | Score: {item.get('final_score')}")
        
        print("\n--- Output Files Exported ---")
        print(f"1. Unified Scene Graph: {session_path}/scene_graph.json")
        print(f"2. Occupancy Grid matrix: {session_path}/occupancy_grid.json")
        print(f"3. Merged Scene Inventory (V1/V2): {session_path}/merged_output.json")
        print(f"4. Spatial Clearance Audit: {session_path}/spatial_analysis.json")
        print(f"5. Design Style Alignment: {session_path}/design_analysis.json")
        print(f"6. Product Recommendations: {session_path}/commerce_analysis.json")
        print(f"7. Budget Utilization profile: {session_path}/budget_analysis.json")
        print(f"8. Recommendation Rationale: {session_path}/explain_analysis.json")
        print(f"9. Layout Transition Delta: {session_path}/reconstruction_analysis.json")
        print(f"10. Visual Reconstruction Preview: {session_path}/reconstruction_preview.json")

        # Interactive notebook visualization block
        try:
            recon_img_path = pathlib.Path(session_path) / "processed_reconstructed.png"
            if recon_img_path.exists():
                print("\n🖼️ Rendering Reconstructed Room Layout visualization...")
                from IPython.display import display as ipy_display
                from PIL import Image as PILImage
                ipy_display(PILImage.open(recon_img_path))
        except Exception:
            pass
        
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