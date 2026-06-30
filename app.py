import os
import sys
import uuid
import pathlib
import pprint
from PIL import Image
import gradio as gr

# Ensure the root path is in the sys path
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.vision_agent import VisionAgent
from agents.spatial_agent import SpatialAgent
from agents.design_agent import DesignAgent

# Initialize agents
vision_agent = VisionAgent()
spatial_agent = SpatialAgent()
design_agent = DesignAgent()

def run_pipeline(
    image,
    room_type,
    style_preference,
    budget,
    colors_str,
    brands_str,
    retain_bed,
    keep_window,
    wheelchair,
    wfh
):
    if image is None:
        return (
            "Error: Please upload a room image.",
            [], [], [], {}, [], "Please upload an image to start analysis."
        )

    # Convert NumPy array from Gradio to PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    try:
        # 1. Run Vision Agent (Scene Understanding)
        vision_output = vision_agent.analyze_room(
            images=[image],
            room_type=room_type,
            style_preference=style_preference
        )
        
        session_id = vision_output["session_id"]
        session_path = vision_output["metadata"]["session_path"]
        
        # 2. Run Spatial Agent (Clearance & Path checks)
        spatial_output = spatial_agent.analyze_layout(
            room_data=vision_output,
            proposed_furniture=None,
            session_path=session_path
        )
        
        # Parse inputs
        colors = [c.strip() for c in colors_str.split(",") if c.strip()]
        brands = [b.strip() for b in brands_str.split(",") if b.strip()]
        
        constraints = {
            "retain_existing_bed": retain_bed,
            "keep_window_access": keep_window,
            "wheelchair_accessible": wheelchair,
            "work_from_home": wfh
        }
        
        # 3. Run Design Agent (Style recommendations & placements)
        design_output = design_agent.generate_recommendations(
            vision_output=vision_output,
            spatial_output=spatial_output,
            budget=budget,
            preferred_style=style_preference,
            preferred_colors=colors,
            preferred_brands=brands,
            constraints=constraints,
            session_path=session_path
        )
        
        # Extract fields
        theme = design_output.get("design_theme", "N/A")
        palette = design_output.get("color_palette", [])
        recs = design_output.get("recommended_furniture", [])
        layout = design_output.get("layout_plan", [])
        breakdown = design_output.get("budget_breakdown", {})
        shopping = design_output.get("shopping_requirements", [])
        explanations = "\n".join([f"- {exp}" for exp in design_output.get("design_explanations", [])])
        
        return theme, palette, recs, layout, breakdown, shopping, explanations
        
    except Exception as e:
        return (
            f"Pipeline Error: {str(e)}",
            [], [], [], {}, [], f"Error running agents: {str(e)}"
        )

def run_cli_demo():
    print("==================================================")
    print("Running CLI Pipeline Demo mode...")
    print("Creating dummy room image...")
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    temp_img_path = "temp_room_view.jpg"
    img.save(temp_img_path)
    
    try:
        theme, palette, recs, layout, breakdown, shopping, explanations = run_pipeline(
            image=img,
            room_type="bedroom",
            style_preference="scandinavian",
            budget=50000,
            colors_str="white, oak, light_grey",
            brands_str="IKEA, Urban Ladder",
            retain_bed=True,
            keep_window=True,
            wheelchair=False,
            wfh=True
        )
        print("Pipeline Execution Completed!")
        print(f"Theme: {theme}")
        print(f"Explanations: {explanations}")
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

def launch_web_app():
    # Built using clean, modern, dark interface aesthetics
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate")) as demo:
        gr.Markdown(
            """
            # 🌌 VisionSpace AI - Multi-Agent Planner
            *Convert simple space snapshots into professional, accessible, and budget-optimized interior design plans.*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Room Image")
                input_image = gr.Image(label="Upload Room Photo", type="numpy")
                
                gr.Markdown("### ⚙️ Preferences & Constraints")
                room_type = gr.Dropdown(
                    label="Room Type",
                    choices=["bedroom", "living_room", "dining_room", "study"],
                    value="bedroom"
                )
                style_preference = gr.Dropdown(
                    label="Style Target",
                    choices=["scandinavian", "modern", "minimalist", "industrial", "japandi", "luxury", "bohemian", "traditional", "contemporary"],
                    value="scandinavian"
                )
                budget = gr.Number(label="Total Budget Limit (INR)", value=50000)
                colors_str = gr.Textbox(label="Colors (comma separated)", value="white, oak, light_grey")
                brands_str = gr.Textbox(label="Preferred Brands (comma separated)", value="IKEA, Urban Ladder")
                
                with gr.Row():
                    retain_bed = gr.Checkbox(label="Retain Existing Bed", value=True)
                    keep_window = gr.Checkbox(label="Keep Window Access", value=True)
                with gr.Row():
                    wheelchair = gr.Checkbox(label="Wheelchair Accessible", value=False)
                    wfh = gr.Checkbox(label="Work From Home setup", value=True)
                
                submit_btn = gr.Button("🚀 Run Multi-Agent Optimizer", variant="primary")
                
            with gr.Column(scale=1.2):
                gr.Markdown("### 💡 Recommended Design Plan")
                
                theme_output = gr.Textbox(label="Design Theme")
                
                with gr.Tab("🎨 Color Palette & Explanations"):
                    palette_output = gr.JSON(label="Colors")
                    explanations_output = gr.Markdown(label="Style & Placement Explanations")
                    
                with gr.Tab("🛏️ Recommended Furniture & Layout"):
                    recs_output = gr.JSON(label="Recommendations Sheet")
                    layout_output = gr.JSON(label="Layout Coordinates Plan")
                    
                with gr.Tab("🛍️ Budget & Shopping Requirements"):
                    breakdown_output = gr.JSON(label="Budget Category Allocation")
                    shopping_output = gr.JSON(label="Commerce Requirements Specification")

        submit_btn.click(
            fn=run_pipeline,
            inputs=[
                input_image, room_type, style_preference, budget,
                colors_str, brands_str, retain_bed, keep_window,
                wheelchair, wfh
            ],
            outputs=[
                theme_output, palette_output, recs_output,
                layout_output, breakdown_output, shopping_output,
                explanations_output
            ]
        )
        
    # On Kaggle, share=True enables a public shareable link
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    demo.launch(share=is_kaggle, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_demo()
    else:
        launch_web_app()