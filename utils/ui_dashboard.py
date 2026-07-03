"""Interactive Notebook UI Dashboard Module for VisionSpace AI.

Constructs HTML widgets and control panels within Kaggle environments to configure
Phase 0 preferences, trigger pipeline execution, and render layout delta visualizations.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional

import ipywidgets as widgets
from IPython.display import HTML, display
from PIL import Image

from workflow.langgraph_workflow import ProductionPipelineV2

logger = logging.getLogger("VisionSpaceAI.UIDashboard")


def image_to_base64(img_path: pathlib.Path) -> str:
    """Encodes PIL image file directly to base64 inline string."""
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return ""


def display_interactive_dashboard(uploader_widget: Optional[Any] = None) -> None:
    """Constructs and renders the interactive dashboard in a Jupyter/Kaggle cell."""
    
    # Instantiate FileUpload widget natively if not passed
    if uploader_widget is None:
        uploader_widget = widgets.FileUpload(
            accept='image/*',
            multiple=True,
            description="📸 Upload Room Snaps",
            layout=widgets.Layout(width="250px", height="35px")
        )

    # 1. Inputs control form widgets
    style_widget = widgets.Dropdown(
        options=["modern", "minimalist", "scandinavian", "contemporary", "industrial"],
        value="modern",
        description="Style Theme:"
    )
    
    budget_widget = widgets.IntSlider(
        value=40000,
        min=5000,
        max=200000,
        step=5000,
        description="Max Budget:"
    )
    
    brands_widget = widgets.SelectMultiple(
        options=["IKEA", "Urban Ladder", "Godrej Interio", "Green Soul"],
        value=["IKEA", "Godrej Interio", "Green Soul"],
        description="Brands:"
    )
    
    products_widget = widgets.Text(
        value="study table, office chair, bookshelf",
        placeholder="e.g. study desk, reading chair, lamp",
        description="Required Items:"
    )
    
    run_btn = widgets.Button(
        description=" Run Production Pipeline",
        button_style="success",
        layout=widgets.Layout(width="250px", height="40px", margin="20px 0px")
    )
    
    # Main outputs container
    output_area = widgets.Output()
    
    # State tracking dictionary
    state = {
        "pipeline": None,
        "out": None,
        "session_path": None,
        "image_box": None
    }

    # Layout forms HBox/VBox
    form_layout = widgets.VBox([
        widgets.HTML("<h3> Configure Workspace Selections (Phase 0)</h3>"),
        widgets.HTML("<b>1. Upload Room Snaps (Select one or more images):</b>"),
        uploader_widget,
        widgets.HTML("<br><b>2. Set Curation & Aesthetic Parameters:</b>"),
        style_widget,
        budget_widget,
        brands_widget,
        products_widget,
        run_btn
    ], layout=widgets.Layout(padding="15px", border="1px solid #ccc", border_radius="8px", width="500px"))

    display(form_layout)
    display(output_area)

    def get_comparison_html(b64_before: str, b64_after: str) -> str:
        return f"""
        <div class="image-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div class="image-card" style="background: #1a202c; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; text-align: center;">
                <img src="data:image/png;base64,{b64_before}" alt="Before View" style="max-width: 100%; border-radius: 6px; height: auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <div style="font-weight: bold; margin-top: 8px; color: #a0aec0;">Original Viewpoint</div>
            </div>
            <div class="image-card" style="background: #1a202c; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; text-align: center;">
                <img src="data:image/png;base64,{b64_after}" alt="After View" style="max-width: 100%; border-radius: 6px; height: auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <div style="font-weight: bold; margin-top: 8px; color: #a0aec0;">Reconstructed Layout Preview</div>
            </div>
        </div>
        """

    def on_try_product(product_item: Dict[str, Any]) -> None:
        if not state["pipeline"] or not state["session_path"]:
            return
        
        with output_area:
            print(f"🛋️ Try in Room: Placing '{product_item.get('name')}' in layout...")
            try:
                # Dynamically rerun reconstruction using only the selected product
                recon = state["pipeline"].reconstruction_agent.plan_reconstruction(
                    vision_output=state["out"]["vision"]["merged_scene"],
                    commerce_output={"recommended_products": [product_item]},
                    session_path=state["session_path"]
                )
                
                # Fetch image paths
                before_img_name = state["out"].get("reconstruction", {}).get("before_image", "original_1.png")
                after_img_name = recon.get("after_image", "processed_reconstructed.png")
                
                before_path = state["session_path"] / "original_images" / before_img_name
                if not before_path.exists():
                    before_path = state["session_path"] / "processed_images" / "processed_1.png"
                after_path = state["session_path"] / after_img_name
                
                b64_before = image_to_base64(before_path)
                b64_after = image_to_base64(after_path)
                
                # Update the HTML comparison image box
                state["image_box"].value = get_comparison_html(b64_before, b64_after)
                print(f" Success! Placed {product_item.get('name')} in the preview layout.")
            except Exception as e:
                print(f" Product placement failed: {str(e)}")

    def on_run_clicked(b: Any) -> None:
        with output_area:
            output_area.clear_output()
            print(" Checking uploaded images...")
            
            # Extract uploaded images
            uploaded_files = uploader_widget.value
            images = []
            
            # Clean and recreate uploads dir
            uploads_dir = pathlib.Path("uploads")
            if uploads_dir.exists():
                shutil.rmtree(uploads_dir)
            uploads_dir.mkdir(parents=True, exist_ok=True)

            if uploaded_files:
                if isinstance(uploaded_files, dict):
                    for file_name, file_info in uploaded_files.items():
                        content = file_info['content']
                        img = Image.open(io.BytesIO(content))
                        images.append(img)
                        # Save to uploads dir
                        with open(uploads_dir / file_name, "wb") as f:
                            f.write(content)
                else:
                    for file_info in uploaded_files:
                        file_name = file_info['name']
                        content = file_info['content']
                        img = Image.open(io.BytesIO(content))
                        images.append(img)
                        # Save to uploads dir
                        with open(uploads_dir / file_name, "wb") as f:
                            f.write(content)
                print(f" Loaded {len(images)} uploaded images.")
            else:
                # Fallback check
                print("ℹ No widget uploads detected. Searching fallback room view assets...")
                fallback_paths = ["assets/front-view.png", "assets/left-view.png", "assets/right-view.png"]
                for p in fallback_paths:
                    path = pathlib.Path(p)
                    if path.exists():
                        images.append(Image.open(path))
                        # Save to uploads
                        shutil.copy(path, uploads_dir / path.name)
                        
            if not images:
                print(" Error: No room images found! Please upload files first.")
                return

            print(" Instantiating orchestrator...")
            pipeline = ProductionPipelineV2()

            # Compile Phase 0 preferences
            user_prefs = {
                "required_products": [p.strip() for p in products_widget.value.split(",") if p.strip()],
                "preferred_brands": list(brands_widget.value),
                "excluded_brands": [],
                "budget_range": {
                    "minimum": 0,
                    "maximum": budget_widget.value
                },
                "preferred_style": style_widget.value,
                "color_preferences": ["white", "wood"]
            }

            print("\n Executing sequential pipeline workflow (Phases 1-7)...")
            try:
                out = pipeline.run_pipeline(
                    images=images,
                    user_preferences=user_prefs,
                    room_type="bedroom",
                    progress_callback=lambda step: print(f"  [Step] {step}")
                )
                
                session_path = pathlib.Path(out["session_path"])
                state["pipeline"] = pipeline
                state["out"] = out
                state["session_path"] = session_path

                # Fetch results file paths
                before_img_name = out.get("reconstruction", {}).get("before_image", "original_1.png")
                after_img_name = out.get("reconstruction", {}).get("after_image", "processed_reconstructed.png")
                
                before_path = session_path / "original_images" / before_img_name
                if not before_path.exists():
                    before_path = session_path / "processed_images" / "processed_1.png"
                after_path = session_path / after_img_name

                # Load image base64 elements
                b64_before = image_to_base64(before_path)
                b64_after = image_to_base64(after_path)

                # Initialize HTML comparison widget
                state["image_box"] = widgets.HTML(value=get_comparison_html(b64_before, b64_after))

                # Render dynamic styled dashboard
                dashboard_details = f"""
                <style>
                    .dashboard-container {{
                        font-family: 'Inter', system-ui, -apple-system, sans-serif;
                        color: #ffffff;
                        background: #121824;
                        padding: 20px;
                        border-radius: 12px;
                        margin-top: 20px;
                    }}
                    .dashboard-header {{
                        text-align: center;
                        margin-bottom: 25px;
                        border-bottom: 2px solid #2d3748;
                        padding-bottom: 10px;
                    }}
                    .details-grid {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                        margin-bottom: 20px;
                    }}
                    .info-card {{
                        background: #1a202c;
                        border-radius: 8px;
                        padding: 15px;
                        border: 1px solid #2d3748;
                    }}
                    .info-card h4 {{
                        margin-top: 0;
                        color: #38bdf8;
                        border-bottom: 1px solid #2d3748;
                        padding-bottom: 5px;
                    }}
                    .score-tag {{
                        background: #1e293b;
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-size: 12px;
                        color: #fbbf24;
                    }}
                </style>
                <div class="dashboard-container">
                    <div class="dashboard-header">
                        <h2> VisionSpace AI Layout Reconstruction Dashboard</h2>
                        <p style="color: #94a3b8;">Interactive View Comparison and Product Curation Summary</p>
                    </div>
                    
                    <div class="details-grid">
                        <div class="info-card">
                            <h4> Spatial Audit & Design Alignment</h4>
                            <p><b>Walkability Index:</b> {out['spatial'].get('walkability_score')}/100</p>
                            <p><b>Style Theme Identified:</b> {out['design'].get('design_theme', 'modern minimal')}</p>
                            <p><b>Usable Floor Area:</b> {out['spatial'].get('usable_area', 0)} sqm</p>
                        </div>
                        
                        <div class="info-card">
                            <h4> Budget Breakdown</h4>
                            <p><b>Total Selected Cost:</b> Rs. {out['budget'].get('total')}</p>
                            <p><b>Limit Setting:</b> Rs. {out['budget'].get('budget_limit')}</p>
                            <p><b>Reserves Remaining:</b> Rs. {out['budget'].get('remaining_budget')}</p>
                        </div>
                    </div>
                </div>
                """
                display(HTML(dashboard_details))
                display(state["image_box"])

                # Render dynamic catalog cards with "Try in Room" callback buttons
                print("\n Recommended Products List (Click '🛋️ Place in Room' to test in layout):")
                
                recommended_list = out["commerce"].get("recommended_products", [])
                
                for item in recommended_list[:10]:
                    img_url = item.get("image") or "https://via.placeholder.com/150"
                    
                    # Create card details layout
                    card_html = f"""
                    <div style="display: flex; align-items: center; background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 12px; margin-bottom: 10px; width: 680px; color: #ffffff;">
                        <img src="{img_url}" style="width: 80px; height: 80px; border-radius: 6px; object-fit: cover; margin-right: 15px; border: 1px solid #2d3748;">
                        <div style="flex-grow: 1;">
                            <div style="font-weight: bold; font-size: 16px; color: #38bdf8;">{item.get('name')}</div>
                            <div style="color: #94a3b8; font-size: 13px; margin: 4px 0;">Brand: <b>{item.get('brand')}</b> | Style: {item.get('style')}</div>
                            <div>
                                <span style="font-weight: bold; color: #fbbf24;">Rs. {item.get('price')}</span>
                                <span class="score-tag" style="margin-left: 10px;">Rank Score: {item.get('final_score')}</span>
                            </div>
                        </div>
                    </div>
                    """
                    
                    try_btn = widgets.Button(
                        description="🛋️ Place in Room",
                        button_style="info",
                        layout=widgets.Layout(width="150px", height="35px", margin="auto 10px")
                    )
                    
                    # Bind local callback
                    try_btn.on_click(lambda b, p=item: on_try_product(p))
                    
                    # Display card and button row
                    display(widgets.HBox([widgets.HTML(card_html), try_btn]))

            except Exception as e:
                print(f" Execution crashed: {str(e)}")
                raise e

    run_btn.on_click(on_run_clicked)
