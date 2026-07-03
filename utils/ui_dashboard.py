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


def display_interactive_dashboard(uploader_widget: Any) -> None:
    """Constructs and renders the interactive dashboard in a Jupyter/Kaggle cell."""
    
    # 1. Inputs control form widgets
    style_widget = widgets.Dropdown(
        options=["modern", "minimalist", "scandinavian", "contemporary", "industrial"],
        value="modern",
        description="Style Theme:"
    )
    
    budget_widget = widgets.IntSlider(
        value=50000,
        min=5000,
        max=200000,
        step=5000,
        description="Max Budget:"
    )
    
    brands_widget = widgets.SelectMultiple(
        options=["IKEA", "Urban Ladder", "Godrej Interio"],
        value=["IKEA", "Urban Ladder"],
        description="Brands:"
    )
    
    products_widget = widgets.Text(
        value="study table, office chair",
        placeholder="e.g. study desk, reading chair, lamp",
        description="Required Items:"
    )
    
    run_btn = widgets.Button(
        description="🚀 Run Production Pipeline",
        button_style="success",
        layout=widgets.Layout(width="250px", height="40px", margin="20px 0px")
    )
    
    output_area = widgets.Output()

    # Layout forms HBox/VBox
    form_layout = widgets.VBox([
        widgets.HTML("<h3>🛠️ Step 2: Configure Workspace Selections (Phase 0)</h3>"),
        style_widget,
        budget_widget,
        brands_widget,
        products_widget,
        run_btn
    ], layout=widgets.Layout(padding="15px", border="1px solid #ccc", border_radius="8px", width="500px"))

    display(form_layout)
    display(output_area)

    def on_run_clicked(b: Any) -> None:
        with output_area:
            output_area.clear_output()
            print("⏳ Checking uploaded images...")
            
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
                print(f"✅ Loaded {len(images)} uploaded images from Cell 2 widget.")
            else:
                # Fallback check
                print("ℹ️ No widget uploads detected. Searching fallback room view assets...")
                fallback_paths = ["assets/front-view.png", "assets/left-view.png", "assets/right-view.png"]
                for p in fallback_paths:
                    path = pathlib.Path(p)
                    if path.exists():
                        images.append(Image.open(path))
                        # Save to uploads
                        shutil.copy(path, uploads_dir / path.name)
                        
            if not images:
                print("❌ Error: No room images found! Please select files in Cell 2 first.")
                return

            print("🤖 Instantiating orchestrator...")
            pipeline = ProductionPipelineV2()

            # Compile Phase 0 preferences
            user_prefs = {
                "required_products": [p.strip() for p in products_widget.value.split(",") if p.strip()],
                "preferred_brands": list(brands_widget.value),
                "excluded_brands": ["Local"],
                "budget_range": {
                    "minimum": 0,
                    "maximum": budget_widget.value
                },
                "preferred_style": style_widget.value,
                "color_preferences": ["white", "wood"]
            }

            print("\n🚀 Executing sequential pipeline workflow (Phases 1-7)...")
            try:
                out = pipeline.run_pipeline(
                    images=images,
                    user_preferences=user_prefs,
                    room_type="bedroom",
                    progress_callback=lambda step: print(f"  [Step] {step}")
                )
                
                # Fetch results file paths
                session_path = pathlib.Path(out["session_path"])
                before_img_name = out.get("reconstruction", {}).get("before_image", "original_1.png")
                after_img_name = out.get("reconstruction", {}).get("after_image", "processed_reconstructed.png")
                
                before_path = session_path / "original_images" / before_img_name
                if not before_path.exists():
                    before_path = session_path / "processed_images" / "processed_1.png"
                after_path = session_path / after_img_name

                # Load image base64 elements
                b64_before = image_to_base64(before_path)
                b64_after = image_to_base64(after_path)

                # Render dynamic styled dashboard
                html_code = f"""
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
                    .image-grid {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .image-card {{
                        background: #1a202c;
                        border-radius: 8px;
                        padding: 10px;
                        border: 1px solid #2d3748;
                        text-align: center;
                    }}
                    .image-card img {{
                        max-width: 100%;
                        border-radius: 6px;
                        height: auto;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                    }}
                    .image-label {{
                        font-weight: bold;
                        margin-top: 8px;
                        color: #a0aec0;
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
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }}
                    th, td {{
                        text-align: left;
                        padding: 8px;
                        border-bottom: 1px solid #2d3748;
                    }}
                    th {{
                        color: #94a3b8;
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
                        <h2>🌌 VisionSpace AI Layout Reconstruction Dashboard</h2>
                        <p style="color: #94a3b8;">Interactive View Comparison and Product Curation Summary</p>
                    </div>
                    
                    <div class="image-grid">
                        <div class="image-card">
                            <img src="data:image/png;base64,{b64_before}" alt="Before Layout">
                            <div class="image-label">Original Viewpoint</div>
                        </div>
                        <div class="image-card">
                            <img src="data:image/png;base64,{b64_after}" alt="Reconstructed Layout">
                            <div class="image-label">Visual Layout Preview (Product Placed Overlay)</div>
                        </div>
                    </div>

                    <div class="details-grid">
                        <div class="info-card">
                            <h4>📋 Spatial Audit & Design Alignment</h4>
                            <p><b>Walkability Index:</b> {out['spatial'].get('walkability_score')}/100</p>
                            <p><b>Style Style Identified:</b> {out['design'].get('design_theme', out['design'].get('current_style', 'modern'))}</p>
                            <p><b>Usable Floor Area:</b> {out['spatial'].get('usable_area', out['spatial'].get('usable_area_percentage', 0))} sqm</p>
                            <p><b>Improvement Opportunities:</b></p>
                            <ul>
                                {"".join(f"<li>{item}</li>" for item in out['design'].get('improvement_opportunities', [f"Applied design theme: {out['design'].get('design_theme', 'modern minimal')}"]))}
                            </ul>
                        </div>
                        
                        <div class="info-card">
                            <h4>💰 Budget Breakdown</h4>
                            <p><b>Total Recommended Cost:</b> Rs. {out['budget'].get('total', out['budget'].get('total_cost', 0))}</p>
                            <p><b>Limit Setting:</b> Rs. {out['budget'].get('budget_limit')}</p>
                            <p><b>Reserves Remaining:</b> Rs. {out['budget'].get('remaining_budget')}</p>
                            <p><b>Spending Utilization:</b> {out['budget'].get('budget_utilization_percentage', int((out['budget'].get('total', 0) / max(1, out['budget'].get('budget_limit', 1))) * 100))}%</p>
                        </div>
                    </div>

                    <div class="info-card">
                        <h4>🛒 Dynamic Shopping Curation (Google Shopping Live Ranked Results)</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>Item</th>
                                    <th>Brand</th>
                                    <th>Price</th>
                                    <th>Scoring Parameters (Fit / Style / Budget)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join(f"<tr><td>{p.get('name')}</td><td>{p.get('brand')}</td><td>Rs. {p.get('price')}</td><td><span class='score-tag'>Rank Score: {p.get('final_score')}</span></td></tr>" for p in out['commerce'].get('recommended_products', []))}
                            </tbody>
                        </table>
                    </div>
                </div>
                """
                display(HTML(html_code))
                
            except Exception as e:
                print(f"❌ Execution crashed: {str(e)}")
                raise e

    run_btn.on_click(on_run_clicked)
