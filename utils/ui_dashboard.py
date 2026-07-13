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


def execute_scripts_from_html(html_str: str) -> None:
    """Extracts and displays script tags from HTML to force evaluation in Jupyter/Kaggle."""
    import re
    from IPython.display import HTML, display
    scripts = re.findall(r'<script>(.*?)</script>', html_str, re.DOTALL)
    for script_content in scripts:
        display(HTML(f"<script>{script_content}</script>"))


def generate_comparison_html(session_path: pathlib.Path, recon: Dict[str, Any]) -> str:
    """Generates the comparison HTML for viewpoints, respecting the strict false-preview guard."""
    reconstruction_meta = recon.get("reconstruction") or {}
    verified_product_ids = reconstruction_meta.get("verified_product_ids") or []
    verified_count = len(verified_product_ids)
    
    reconstructed_views = sorted(list(session_path.glob("processed_reconstructed_*.png")))
    if not reconstructed_views:
        reconstructed_views = [session_path / "processed_reconstructed.png"]
        
    comparison_html_list = []
    
    for idx, after_path in enumerate(reconstructed_views):
        before_img_name = f"original_{idx + 1}.png"
        before_path = session_path / "original_images" / before_img_name
        if not before_path.exists():
            before_path = session_path / "processed_images" / f"processed_{idx + 1}.png"
        if not before_path.exists():
            before_path = session_path / "original_images" / "original_1.png"
        if not before_path.exists():
            before_path = session_path / "processed_images" / "processed_1.png"
            
        # Target path for preview
        preview_path = after_path if (verified_count > 0 and after_path.exists()) else before_path
        
        if before_path.exists() and preview_path.exists():
            b64_before = image_to_base64(before_path)
            view_title = f"Viewpoint {idx + 1}"
            
            natural_w, natural_h = 800, 600
            try:
                with Image.open(before_path) as b_img:
                    natural_w, natural_h = b_img.size
            except Exception:
                pass
            
            product_layers_html = ""
            if verified_count > 0:
                product_results = recon.get("product_results", [])
                for p in product_results:
                    if p.get("feasibility_status") == "valid" and p.get("render_status") == "verified":
                        p_id = p.get("product_id")
                        b64_cutout = p.get("b64_cutout")
                        coords = p.get("coords")
                        if b64_cutout and coords:
                            x1, y1, x2, y2 = coords
                            left = x1 * (500 / natural_w)
                            top = y1 * (375 / natural_h)
                            width = (x2 - x1) * (500 / natural_w)
                            height = (y2 - y1) * (375 / natural_h)
                            
                            product_layers_html += f"""
                            <div class="draggable-product" data-id="{p_id}" data-name="{p.get('name', 'Product')}" style="position: absolute; left: {left:.1f}px; top: {top:.1f}px; width: {width:.1f}px; height: {height:.1f}px; cursor: move; transform: rotate(0deg); transform-origin: center; display: inline-block; z-index: 10; user-select: none; -webkit-user-select: none; -moz-user-select: none; -ms-user-select: none; background: transparent;">
                                <img src="data:image/png;base64,{b64_cutout}" style="width: 100%; height: 100%; pointer-events: none; object-fit: contain; user-select: none; -webkit-user-select: none; -moz-user-select: none;">
                            </div>
                            """
            
            if verified_count > 0:
                header_title = f"Interactive Layout Reconstruction - {view_title}"
                header_color = "#38bdf8"
                after_label = f"Interactive Room Canvas ({view_title})"
                
                right_card_html = f"""
                <div class="image-card" style="background: #1a202c; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <div style="position: relative; width: 500px; height: 375px; background: #000; border-radius: 6px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                        <div id="interactive-room-{idx}" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; background-image: url('data:image/png;base64,{b64_before}'); background-size: 100% 100%; pointer-events: auto; user-select: none; -webkit-user-select: none; -moz-user-select: none;"></div>
                        {product_layers_html}
                        
                        <div id="toolbar-{idx}" style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); background: rgba(26, 32, 44, 0.95); border: 1px solid #4a5568; border-radius: 20px; padding: 4px 12px; display: none; gap: 8px; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.5); align-items: center;">
                            <span id="tool-name-{idx}" style="color: #e2e8f0; font-size: 11px; font-family: sans-serif; font-weight: bold; margin-right: 4px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">Item</span>
                            <button class="btn-rot-l-{idx}" style="background: #2d3748; color: #fff; border: 1px solid #4a5568; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;" title="Rotate Left">↺</button>
                            <button class="btn-rot-r-{idx}" style="background: #2d3748; color: #fff; border: 1px solid #4a5568; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;" title="Rotate Right">↻</button>
                            <button class="btn-scale-d-{idx}" style="background: #2d3748; color: #fff; border: 1px solid #4a5568; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;" title="Scale Down">−</button>
                            <button class="btn-scale-u-{idx}" style="background: #2d3748; color: #fff; border: 1px solid #4a5568; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;" title="Scale Up">+</button>
                            <button class="btn-del-{idx}" style="background: #e53e3e; color: #fff; border: none; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;" title="Remove Item">🗑️</button>
                        </div>
                    </div>
                    <div style="font-weight: bold; margin-top: 8px; color: #a0aec0; font-family: sans-serif;">{after_label}</div>
                    <div style="margin-top: 10px; font-family: sans-serif;">
                        <button class="btn-download-{idx}" style="background-color: #2563eb; color: #ffffff; padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; font-size: 12px;">Download Customized View {idx+1}</button>
                    </div>
                </div>
                """
            else:
                header_title = f"Original Room — No Products Successfully Rendered - {view_title}"
                header_color = "#ef4444"
                after_label = f"Original View (Reconstruction Failed/No Space Available) ({view_title})"
                button_label = f"Download Original View {idx+1}"
                b64_after = image_to_base64(preview_path)
                
                right_card_html = f"""
                <div class="image-card" style="background: #1a202c; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; text-align: center;">
                    <img src="data:image/png;base64,{b64_after}" alt="After View" style="max-width: 100%; border-radius: 6px; height: auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                    <div style="font-weight: bold; margin-top: 8px; color: #a0aec0; font-family: sans-serif;">{after_label}</div>
                    <div style="margin-top: 10px; font-family: sans-serif;">
                        <a href="data:image/png;base64,{b64_after}" download="reconstructed_layout_{idx+1}.png" style="background-color: #2563eb; color: #ffffff; padding: 6px 12px; border-radius: 4px; text-decoration: none; display: inline-block; font-weight: bold; font-size: 12px;">{button_label}</a>
                    </div>
                </div>
                """
            
            # Interactive scripting block injected per viewpoint card
            interactive_script_html = f"""
            <script>
            (function() {{
                // 1. Redefine global helper functions every run to keep references fresh
                window.startItemDrag = function(e, el, viewIdx) {{
                    e.preventDefault();
                    e.stopPropagation();
                    
                    if (!window.VisionSpaceState) {{
                        window.VisionSpaceState = {{}};
                    }}
                    if (!window.VisionSpaceState[viewIdx]) {{
                        window.VisionSpaceState[viewIdx] = {{
                            activeElement: null,
                            isDragging: false,
                            startX: 0,
                            startY: 0,
                            startLeft: 0,
                            startTop: 0
                        }};
                    }}
                    
                    const state = window.VisionSpaceState[viewIdx];
                    
                    window.selectElement(el, viewIdx);
                    state.isDragging = true;
                    state.activeElement = el;
                    
                    const isTouch = e.type.startsWith('touch');
                    const clientX = isTouch ? e.touches[0].clientX : e.clientX;
                    const clientY = isTouch ? e.touches[0].clientY : e.clientY;
                    
                    state.startX = clientX;
                    state.startY = clientY;
                    state.startLeft = parseFloat(el.style.left) || 0;
                    state.startTop = parseFloat(el.style.top) || 0;
                }};
                
                window.roomBackgroundClicked = function(e, viewIdx) {{
                    window.deselectAll(viewIdx);
                }};
                
                window.selectElement = function(el, viewIdx) {{
                    window.deselectAll(viewIdx);
                    
                    if (!window.VisionSpaceState) window.VisionSpaceState = {{}};
                    if (!window.VisionSpaceState[viewIdx]) {{
                        window.VisionSpaceState[viewIdx] = {{ activeElement: null }};
                    }}
                    window.VisionSpaceState[viewIdx].activeElement = el;
                    
                    el.style.outline = '2px dashed #2563eb';
                    el.style.boxShadow = '0 0 12px rgba(37, 99, 235, 0.5)';
                    
                    const toolbar = document.getElementById('toolbar-' + viewIdx);
                    const toolName = document.getElementById('tool-name-' + viewIdx);
                    if (toolbar) toolbar.style.display = 'flex';
                    if (toolName) toolName.innerText = el.getAttribute('data-name');
                }};
                
                window.deselectAll = function(viewIdx) {{
                    if (window.VisionSpaceState && window.VisionSpaceState[viewIdx]) {{
                        window.VisionSpaceState[viewIdx].activeElement = null;
                    }}
                    const container = document.getElementById('interactive-room-' + viewIdx);
                    if (container) {{
                        const products = container.parentElement.querySelectorAll('.draggable-product');
                        products.forEach(p => {{
                            p.style.outline = 'none';
                            p.style.boxShadow = 'none';
                        }});
                    }}
                    const toolbar = document.getElementById('toolbar-' + viewIdx);
                    if (toolbar) toolbar.style.display = 'none';
                }};
                
                window.rotateItem = function(viewIdx, deg) {{
                    const state = window.VisionSpaceState ? window.VisionSpaceState[viewIdx] : null;
                    if (state && state.activeElement) {{
                        let currentRotation = 0;
                        const transform = state.activeElement.style.transform || '';
                        const match = transform.match(/rotate\\(([-0-9.]+)deg\\)/);
                        if (match) currentRotation = parseFloat(match[1]);
                        
                        let newRotation = (currentRotation + deg) % 360;
                        state.activeElement.style.transform = `rotate(${{newRotation}}deg)`;
                    }}
                }};
                
                window.scaleItem = function(viewIdx, scale) {{
                    const state = window.VisionSpaceState ? window.VisionSpaceState[viewIdx] : null;
                    if (state && state.activeElement) {{
                        const el = state.activeElement;
                        const currentW = parseFloat(el.style.width);
                        const currentH = parseFloat(el.style.height);
                        const currentL = parseFloat(el.style.left);
                        const currentT = parseFloat(el.style.top);
                        
                        const newW = currentW * scale;
                        const newH = currentH * scale;
                        
                        el.style.width = newW + 'px';
                        el.style.height = newH + 'px';
                        el.style.left = (currentL - (newW - currentW)/2) + 'px';
                        el.style.top = (currentT - (newH - currentH)/2) + 'px';
                    }}
                }};
                
                window.deleteItem = function(viewIdx) {{
                    const state = window.VisionSpaceState ? window.VisionSpaceState[viewIdx] : null;
                    if (state && state.activeElement) {{
                        state.activeElement.style.display = 'none';
                        window.deselectAll(viewIdx);
                    }}
                }};
                
                window.downloadInteractiveLayout = function(viewIdx) {{
                    const container = document.getElementById('interactive-room-' + viewIdx);
                    const bgImgUrl = container.style.backgroundImage.slice(5, -2).replace(/"/g, "");
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    const bg = new Image();
                    bg.src = bgImgUrl;
                    bg.onload = function() {{
                        canvas.width = bg.naturalWidth || 800;
                        canvas.height = bg.naturalHeight || 600;
                        ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
                        
                        const displayW = container.clientWidth;
                        const displayH = container.clientHeight;
                        const scaleX = canvas.width / displayW;
                        const scaleY = canvas.height / displayH;
                        
                        const layers = container.parentElement.querySelectorAll('.draggable-product');
                        layers.forEach(layer => {{
                            if (layer.style.display === 'none') return;
                            const img = layer.querySelector('img');
                            const layerW = parseFloat(layer.style.width) * scaleX;
                            const layerH = parseFloat(layer.style.height) * scaleY;
                            const layerX = (parseFloat(layer.style.left) + parseFloat(layer.style.width)/2) * scaleX;
                            const layerY = (parseFloat(layer.style.top) + parseFloat(layer.style.height)/2) * scaleY;
                            
                            let angle = 0;
                            const transform = layer.style.transform || '';
                            const rotateMatch = transform.match(/rotate\\(([-0-9.]+)deg\\)/);
                            if (rotateMatch) angle = parseFloat(rotateMatch[1]);
                            
                            ctx.save();
                            ctx.translate(layerX, layerY);
                            ctx.rotate(angle * Math.PI / 180);
                            ctx.drawImage(img, -layerW / 2, -layerH / 2, layerW, layerH);
                            ctx.restore();
                        }});
                        
                        const link = document.createElement('a');
                        link.download = 'reconstructed_layout_' + (viewIdx + 1) + '.png';
                        link.href = canvas.toDataURL('image/png');
                        link.click();
                    }};
                }};

                window.bindCustomizerEvents = function(viewIdx) {{
                    const container = document.getElementById('interactive-room-' + viewIdx);
                    if (!container) return;
                    
                    window.deselectAll(viewIdx);
                    
                    const products = container.parentElement.querySelectorAll('.draggable-product');
                    products.forEach(p => {{
                        if (p.getAttribute('data-listeners-bound') === 'true') return;
                        p.setAttribute('data-listeners-bound', 'true');
                        
                        p.addEventListener('mousedown', function(e) {{
                            window.startItemDrag(e, p, viewIdx);
                        }});
                        p.addEventListener('touchstart', function(e) {{
                            window.startItemDrag(e, p, viewIdx);
                        }});
                    }});
                    
                    if (container.getAttribute('data-listeners-bound') !== 'true') {{
                        container.setAttribute('data-listeners-bound', 'true');
                        container.addEventListener('mousedown', function(e) {{
                            window.roomBackgroundClicked(e, viewIdx);
                        }});
                        container.addEventListener('touchstart', function(e) {{
                            window.roomBackgroundClicked(e, viewIdx);
                        }});
                    }}
                    
                    const toolbar = document.getElementById('toolbar-' + viewIdx);
                    if (toolbar && toolbar.getAttribute('data-listeners-bound') !== 'true') {{
                        toolbar.setAttribute('data-listeners-bound', 'true');
                        const btnRotL = toolbar.querySelector('.btn-rot-l-' + viewIdx);
                        const btnRotR = toolbar.querySelector('.btn-rot-r-' + viewIdx);
                        const btnScaleD = toolbar.querySelector('.btn-scale-d-' + viewIdx);
                        const btnScaleU = toolbar.querySelector('.btn-scale-u-' + viewIdx);
                        const btnDel = toolbar.querySelector('.btn-del-' + viewIdx);
                        
                        if (btnRotL) btnRotL.addEventListener('click', function() {{ window.rotateItem(viewIdx, -15); }});
                        if (btnRotR) btnRotR.addEventListener('click', function() {{ window.rotateItem(viewIdx, 15); }});
                        if (btnScaleD) btnScaleD.addEventListener('click', function() {{ window.scaleItem(viewIdx, 0.9); }});
                        if (btnScaleU) btnScaleU.addEventListener('click', function() {{ window.scaleItem(viewIdx, 1.1); }});
                        if (btnDel) btnDel.addEventListener('click', function() {{ window.deleteItem(viewIdx); }});
                    }}
                    
                    const card = container.parentElement;
                    const btnDownload = card.querySelector('.btn-download-' + viewIdx);
                    if (btnDownload && btnDownload.getAttribute('data-listeners-bound') !== 'true') {{
                        btnDownload.setAttribute('data-listeners-bound', 'true');
                        btnDownload.addEventListener('click', function() {{
                            window.downloadInteractiveLayout(viewIdx);
                        }});
                    }}
                }};

                // 2. Initialize document-level event listeners EXACTLY ONCE
                if (!window.VisionSpaceListenersInitialized) {{
                    window.VisionSpaceListenersInitialized = true;
                    window.VisionSpaceState = {{}};
                    
                    document.addEventListener('mousemove', function(e) {{
                        if (window.VisionSpaceState) {{
                            for (const viewIdx in window.VisionSpaceState) {{
                                const state = window.VisionSpaceState[viewIdx];
                                if (state && state.isDragging && state.activeElement) {{
                                    const dx = e.clientX - state.startX;
                                    const dy = e.clientY - state.startY;
                                    state.activeElement.style.left = (state.startLeft + dx) + 'px';
                                    state.activeElement.style.top = (state.startTop + dy) + 'px';
                                }}
                            }}
                        }}
                    }}, {{ passive: true }});
                    
                    document.addEventListener('mouseup', function() {{
                        if (window.VisionSpaceState) {{
                            for (const viewIdx in window.VisionSpaceState) {{
                                if (window.VisionSpaceState[viewIdx]) {{
                                    window.VisionSpaceState[viewIdx].isDragging = false;
                                }}
                            }}
                        }}
                    }});
                    
                    document.addEventListener('touchmove', function(e) {{
                        if (window.VisionSpaceState) {{
                            for (const viewIdx in window.VisionSpaceState) {{
                                const state = window.VisionSpaceState[viewIdx];
                                if (state && state.isDragging && state.activeElement) {{
                                    const touch = e.touches[0];
                                    const dx = touch.clientX - state.startX;
                                    const dy = touch.clientY - state.startY;
                                    state.activeElement.style.left = (state.startLeft + dx) + 'px';
                                    state.activeElement.style.top = (state.startTop + dy) + 'px';
                                }}
                            }}
                        }}
                    }}, {{ passive: true }});
                    
                    document.addEventListener('touchend', function() {{
                        if (window.VisionSpaceState) {{
                            for (const viewIdx in window.VisionSpaceState) {{
                                if (window.VisionSpaceState[viewIdx]) {{
                                    window.VisionSpaceState[viewIdx].isDragging = false;
                                }}
                            }}
                        }}
                    }});
                    
                    document.addEventListener('dragstart', function(e) {{
                        if (e.target.closest('.draggable-product')) {{
                            e.preventDefault();
                        }}
                    }});
                }}

                // 3. Register Mutation Observer to automatically bind events on dynamic IPyWidgets updates
                if (!window.VisionSpaceObserverInitialized) {{
                    window.VisionSpaceObserverInitialized = true;
                    
                    const observer = new MutationObserver(function(mutations) {{
                        for (let mutation of mutations) {{
                            for (let node of mutation.addedNodes) {{
                                if (node.nodeType === Node.ELEMENT_NODE) {{
                                    const containers = node.querySelectorAll ? node.querySelectorAll('[id^="interactive-room-"]') : [];
                                    containers.forEach(container => {{
                                        const idx = parseInt(container.id.replace('interactive-room-', ''));
                                        window.bindCustomizerEvents(idx);
                                    }});
                                    if (node.id && node.id.startsWith('interactive-room-')) {{
                                        const idx = parseInt(node.id.replace('interactive-room-', ''));
                                        window.bindCustomizerEvents(idx);
                                    }}
                                }}
                            }}
                        }}
                    }});
                    
                    observer.observe(document.body, {{ childList: true, subtree: true }});
                }}

                // 4. Bind events to the current viewport container immediately
                window.bindCustomizerEvents({idx});
            }})();
            </script>
            """
            
            comparison_html = f"""
            <div style="margin-top: 15px; border-top: 1px solid #2d3748; padding-top: 15px;">
                <h4 style="color: {header_color}; font-family: 'Inter', sans-serif; margin-bottom: 10px;">{header_title}</h4>
                <div class="image-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div class="image-card" style="background: #1a202c; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <img src="data:image/png;base64,{b64_before}" alt="Before View" style="max-width: 100%; border-radius: 6px; height: auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                        <div style="font-weight: bold; margin-top: 8px; color: #a0aec0; font-family: sans-serif;">Original Viewpoint ({view_title})</div>
                    </div>
                    {right_card_html}
                </div>
                {interactive_script_html}
            </div>
            """
            comparison_html_list.append(comparison_html)
            
    if verified_count == 0:
        reasons_html = """
        <div style="background-color: #fef2f2; border: 1px solid #fee2e2; border-left: 5px solid #ef4444; border-radius: 6px; padding: 15px; margin-top: 20px; font-family: 'Inter', sans-serif;">
            <h5 style="color: #991b1b; margin: 0 0 10px 0; font-size: 16px;">❌ No selected product was successfully rendered.</h5>
            <ul style="color: #7f1d1d; margin: 0; padding-left: 20px; font-size: 14px; line-height: 1.5;">
        """
        product_results = recon.get("product_results", [])
        for result in product_results:
            reasons_html += f"<li><strong>{result.get('name', 'Product')}</strong>: {result.get('message', 'Failed verification')}</li>"
        reasons_html += """
            </ul>
        </div>
        """
        comparison_html_list.append(reasons_html)
        
    return "\n".join(comparison_html_list)


def display_interactive_dashboard(uploader_widget: Optional[Any] = None) -> None:
    """Constructs and renders the interactive dashboard in a Jupyter/Kaggle cell."""
    
    # Instantiate FileUpload widget natively if not passed
    if uploader_widget is None:
        uploader_widget = widgets.FileUpload(
            accept='image/*',
            multiple=True,
            description=" Upload Room Snaps",
            layout=widgets.Layout(width="250px", height="35px")
        )

    # 1. Inputs control form widgets
    style_widget = widgets.Dropdown(
        options=["modern", "minimalist", "scandinavian", "contemporary", "industrial"],
        value="modern",
        description="Style Theme:"
    )
    
    budget_widget = widgets.IntText(
        value=40000,
        description="Max Budget (Rs):"
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
    
    # State tracking dictionary conforming to exact product placement rules
    state = {
        "pipeline": None,
        "out": None,
        "session_path": None,
        "image_box": None,
        "error_banner": None,
        "selected_products": [],      # User selected try-in-room items
        "approved_placements": [],     # Passed geometry check
        "no_space_products": [],       # Rejections
        "rendered_products": [],       # Visual verification pass
        "buttons": {}                  # Map of product_id -> Button widget
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

    state["budget_widget"] = widgets.HTML()
    state["error_banner"] = widgets.HTML()

    def update_budget_display() -> None:
        if not state["out"]:
            return
        max_budget = float(state["out"]['budget'].get('budget_limit', 40000.0))
        placed_items = state["selected_products"]
        total_placed_cost = sum(float(item.get("price", 0)) for item in placed_items)
        reserves_remaining = max_budget - total_placed_cost
        
        warning_msg = ""
        if reserves_remaining < 0:
            warning_msg = f'<div style="color: #ef4444; font-size: 11px; margin-top: 8px; font-weight: bold;">⚠️ Warning: Selected layout exceeds budget limit by Rs. {abs(reserves_remaining):.2f}!</div>'
            
        # Check if any recommended category options exceed the user budget
        too_expensive_cats = []
        rec_products = state["out"].get("commerce", {}).get("recommended_products", [])
        for item in rec_products:
            if float(item.get("price", 0)) > max_budget:
                cat = item.get("category", "")
                if cat and cat not in too_expensive_cats:
                    too_expensive_cats.append(cat)
        if too_expensive_cats:
            warning_msg += f'<div style="color: #f59e0b; font-size: 11px; margin-top: 5px;">⚠️ Note: Budget limits exclude affordable matches for: {", ".join(too_expensive_cats)}. Displaying premium fallbacks.</div>'
        
        budget_html = f"""
        <div class="info-card" style="width: 320px; margin-bottom: 0;">
            <h4 style="margin-top: 0; color: #38bdf8; border-bottom: 1px solid #2d3748; padding-bottom: 5px;"> Budget Breakdown</h4>
            <p style="margin: 6px 0; font-size: 13px;"><b>Total Selected Cost:</b> Rs. {total_placed_cost:.2f}</p>
            <p style="margin: 6px 0; font-size: 13px;"><b>Limit Setting:</b> Rs. {max_budget:.2f}</p>
            <p style="margin: 6px 0; font-size: 13px;"><b>Reserves Remaining:</b> <span style="color: {'#10b981' if reserves_remaining >= 0 else '#ef4444'}; font-weight: bold;">Rs. {reserves_remaining:.2f}</span></p>
            {warning_msg}
        </div>
        """
        state["budget_widget"].value = budget_html

    def on_try_product(product_item: Dict[str, Any], button: Any) -> None:
        if not state["pipeline"] or not state["session_path"]:
            return
        
        with output_area:
            p_id = product_item.get("product_id")
            selected_ids = [p.get("product_id") for p in state["selected_products"] if p.get("product_id")]
            
            if p_id in selected_ids:
                # Remove it
                state["selected_products"] = [p for p in state["selected_products"] if p.get("product_id") != p_id]
                state["approved_placements"] = [p for p in state["approved_placements"] if p.get("product_id") != p_id]
                state["no_space_products"] = [p for p in state["no_space_products"] if p.get("product_id") != p_id]
                state["rendered_products"] = [p for p in state["rendered_products"] if p.get("product_id") != p_id]
                
                button.description = " Try in Room"
                button.button_style = "info"
                print(f" Removed '{product_item.get('name')}' from selected list.")
            else:
                # Add to selection
                state["selected_products"].append(product_item)
                button.description = " Checking Space..."
                button.button_style = "warning"
                print(f" Selected '{product_item.get('name')}' - validating clearance space...")
                
            try:
                # Dynamically rerun reconstruction using all selected products
                recon = state["pipeline"].reconstruction_agent.plan_reconstruction(
                    vision_output=state["out"]["vision"]["merged_scene"],
                    commerce_output={"recommended_products": state["selected_products"]},
                    session_path=state["session_path"],
                    spatial_output=state["out"]["spatial"]
                )
                
                # Re-initialize placement states
                state["approved_placements"] = []
                state["no_space_products"] = []
                state["rendered_products"] = []
                
                product_results = recon.get("product_results", [])
                for result in product_results:
                    item_id = result.get("product_id")
                    f_status = result.get("feasibility_status")
                    p_status = result.get("placement_status")
                    r_status = result.get("render_status")
                    
                    catalog_item = next((p for p in state["selected_products"] if p.get("product_id") == item_id), None)
                    if not catalog_item:
                        continue
                        
                    btn = state["buttons"].get(item_id)
                    
                    if f_status == "no_space":
                        state["no_space_products"].append(catalog_item)
                        if btn:
                            btn.description = " No Space Available"
                            btn.button_style = "danger"
                    elif r_status == "verified":
                        state["rendered_products"].append(catalog_item)
                        state["approved_placements"].append(catalog_item)
                        if btn:
                            btn.description = " Placed in Room"
                            btn.button_style = "success"
                    elif r_status == "failed" or r_status == "identity_verification_failed":
                        state["approved_placements"].append(catalog_item)
                        if btn:
                            btn.description = " Placement Valid — Rendering Failed"
                            btn.button_style = "warning"
                    elif p_status == "approved":
                        state["approved_placements"].append(catalog_item)
                        if btn:
                            btn.description = " Placed in Room"
                            btn.button_style = "success"
                    else:
                        if btn:
                            btn.description = " Try in Room"
                            btn.button_style = "info"
                            
                # Reset deselected buttons
                current_selected_ids = [p.get("product_id") for p in state["selected_products"]]
                for cached_id, btn in state["buttons"].items():
                    if cached_id not in current_selected_ids:
                        btn.description = " Try in Room"
                        btn.button_style = "info"
                
                # Generate comparison HTML respecting false-preview guard
                comparison_html_val = generate_comparison_html(state["session_path"], recon)
                state["image_box"].value = comparison_html_val
                execute_scripts_from_html(comparison_html_val)
                
                # Check for placement errors
                placement_errors = recon.get("placement_errors", [])
                if placement_errors:
                    error_html_parts = []
                    for err in placement_errors:
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        error_html_parts.append(
                            f'<div style="background-color: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c; padding: 10px; margin-top: 15px; margin-bottom: 5px; border-radius: 4px; font-weight: bold; font-size: 13px; font-family: sans-serif;">⚠️ Placement Warning: {msg}</div>'
                        )
                    state["error_banner"].value = "".join(error_html_parts)
                else:
                    state["error_banner"].value = ""
                
                update_budget_display()
                print(" Success! Updated the layout reconstruction image.")
            except Exception as e:
                print(f" Product placement failed: {str(e)}")

    def on_run_clicked(b: Any) -> None:
        with output_area:
            output_area.clear_output()
            print(" Checking uploaded images...")
            
            # Extract uploaded images
            uploaded_files = uploader_widget.value
            
            # Clear uploader widget to prevent accumulative uploads on subsequent runs
            try:
                uploader_widget.value = ()
            except Exception:
                try:
                    uploader_widget.value = {}
                except Exception:
                    pass
                    
            images = []
            uploads_dir = pathlib.Path("uploads")
            
            if uploaded_files:
                # Clean and recreate uploads dir ONLY when new files are uploaded
                if uploads_dir.exists():
                    shutil.rmtree(uploads_dir)
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
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
                        file_name = file_info.get('name', 'uploaded_image.png')
                        content = file_info['content']
                        img = Image.open(io.BytesIO(content))
                        images.append(img)
                        # Save to uploads dir
                        with open(uploads_dir / file_name, "wb") as f:
                            f.write(content)
                print(f" Loaded {len(images)} uploaded images from widget.")
            else:
                # Check if there are already images in uploads/ from a previous run
                existing_uploads = []
                if uploads_dir.exists():
                    existing_uploads = sorted([
                        p for p in uploads_dir.iterdir()
                        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
                    ])
                
                if existing_uploads:
                    for p in existing_uploads:
                        images.append(Image.open(p))
                    print(f" Loaded {len(images)} previously uploaded images from uploads/ directory.")
                else:
                    # Fallback check
                    print("ℹ No widget uploads detected. Searching fallback room view assets...")
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    # fallback_paths = ["assets/front-view.png", "assets/left-view.png", "assets/right-view.png"]
                    fallback_paths = ["assets/front-view.png"]
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
                state["selected_products"] = list(out.get("selected_products", []))
                state["approved_placements"] = []
                state["no_space_products"] = []
                state["rendered_products"] = []
                state["buttons"] = {}
                recon_data = out.get("reconstruction", {})
                state["image_box"] = widgets.HTML(value=generate_comparison_html(session_path, recon_data))

                # Render dynamic styled dashboard header and spatial audit info
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
                </style>
                <div class="dashboard-container">
                    <div class="dashboard-header">
                        <h2> VisionSpace AI Layout Reconstruction Dashboard</h2>
                        <p style="color: #94a3b8;">Interactive View Comparison and Product Curation Summary</p>
                    </div>
                </div>
                """
                
                spatial_html = f"""
                <style>
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
                </style>
                <div class="info-card" style="width: 320px; margin-bottom: 0;">
                    <h4> Spatial Audit & Design Alignment</h4>
                    <p style="margin: 6px 0; font-size: 13px;"><b>Walkability Index:</b> {out['spatial'].get('walkability_score')}/100</p>
                    <p style="margin: 6px 0; font-size: 13px;"><b>Style Theme Identified:</b> {out['design'].get('design_theme', 'modern minimal')}</p>
                    <p style="margin: 6px 0; font-size: 13px;"><b>Usable Floor Area:</b> {out['spatial'].get('usable_area', 0)} sqm</p>
                </div>
                """
                
                logs_html = f"""
                <style>
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
                </style>
                <div class="info-card" style="margin-top: 15px; margin-bottom: 15px; border-color: #38bdf8;">
                    <h4> Agent Execution Logs & Summaries</h4>
                    <details style="margin-bottom: 8px; cursor: pointer;">
                        <summary style="font-weight: bold; color: #38bdf8;"> Phase 1: Unified Vision Agent Pass</summary>
                        <p style="color: #94a3b8; font-size: 13px; padding-left: 10px; margin-top: 4px;">
                            Loaded room snaps and executed unified Qwen-VL vision pass. Identified room architecture, doors, windows, and existing furniture positions with coordinates.
                        </p>
                    </details>
                    <details style="margin-bottom: 8px; cursor: pointer;">
                        <summary style="font-weight: bold; color: #38bdf8;"> Phase 2 & 3: Spatial clearance & Style Alignment Agent Pass</summary>
                        <p style="color: #94a3b8; font-size: 13px; padding-left: 10px; margin-top: 4px;">
                            Validated layout clearances using spatial clearance logic. Evaluated walkability constraints, generated structural collision paths, and aligned with aesthetic style metrics.
                        </p>
                    </details>
                    <details style="margin-bottom: 8px; cursor: pointer;">
                        <summary style="font-weight: bold; color: #38bdf8;"> Phase 4 & 5: Curation & Budget Curation Agent Pass</summary>
                        <p style="color: #94a3b8; font-size: 13px; padding-left: 10px; margin-top: 4px;">
                            Queried SerpAPI/Mock Catalog to suggest items fitting user specifications. Filtered items to match split budget parameters.
                        </p>
                    </details>
                    <details style="margin-bottom: 8px; cursor: pointer;">
                        <summary style="font-weight: bold; color: #38bdf8;"> Phase 6 & 7: Reconstruction & Output Delta Agent Pass</summary>
                        <p style="color: #94a3b8; font-size: 13px; padding-left: 10px; margin-top: 4px;">
                            Completed generative scene editing. Pasted selected product visuals directly onto the empty zones. Compiled all JSON files and created a ZIP archive.
                        </p>
                    </details>
                </div>
                """
                display(HTML(dashboard_details))
                
                # Initialize dynamic budget display and render spatial alongside it
                update_budget_display()
                display(widgets.HBox([widgets.HTML(spatial_html), state["budget_widget"]], layout=widgets.Layout(gap="20px", margin="10px 0")))
                # Check for placement errors on initial run
                placement_errors = out.get("reconstruction", {}).get("placement_errors", [])
                if placement_errors:
                    error_html_parts = []
                    for err in placement_errors:
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        error_html_parts.append(
                            f'<div style="background-color: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c; padding: 10px; margin-top: 15px; margin-bottom: 5px; border-radius: 4px; font-weight: bold; font-size: 13px; font-family: sans-serif;">⚠️ Placement Warning: {msg}</div>'
                        )
                    state["error_banner"].value = "".join(error_html_parts)
                else:
                    state["error_banner"].value = ""

                display(HTML(logs_html))
                display(state["error_banner"])
                display(state["image_box"])
                execute_scripts_from_html(state["image_box"].value)

                # Render dynamic catalog cards with "Try in Room" callback buttons
                print("\n Recommended Products List (Click ' Place in Room' to test in layout):")
                
                recommended_list = out["commerce"].get("recommended_products", [])
                
                placed_results = out.get("reconstruction", {}).get("product_results", [])
                
                for item in recommended_list:
                    p_id = item.get("product_id")
                    
                    # Set button description and styling dynamically
                    btn_desc = " Try in Room"
                    btn_style = "info"
                    
                    result_item = next((r for r in placed_results if r.get("product_id") == p_id), None)
                    if result_item:
                        f_status = result_item.get("feasibility_status")
                        p_status = result_item.get("placement_status")
                        r_status = result_item.get("render_status")
                        
                        if f_status == "no_space":
                            btn_desc = " No Space Available"
                            btn_style = "danger"
                        elif r_status == "verified":
                            btn_desc = " Placed in Room"
                            btn_style = "success"
                        elif r_status == "failed" or r_status == "identity_verification_failed":
                            btn_desc = " Placement Valid — Rendering Failed"
                            btn_style = "warning"
                        elif p_status == "approved":
                            btn_desc = " Placed in Room"
                            btn_style = "success"
                            
                    # Image priority matching Phase 2 specifications
                    img_url = item.get("thumbnail_url") or item.get("high_resolution_image") or item.get("image") or item.get("image_url") or item.get("thumbnail") or ""
                    
                    if not img_url or "via.placeholder.com" in img_url:
                        cat_lower = item.get("category", "").lower()
                        if "desk" in cat_lower or "table" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?auto=format&fit=crop&w=150&q=80"
                        elif "chair" in cat_lower or "stool" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1505797149-43b0069ec26b?auto=format&fit=crop&w=150&q=80"
                        elif "bookshelf" in cat_lower or "shelf" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1594980596870-8cb524ebf6bc?auto=format&fit=crop&w=150&q=80"
                        elif "bed" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=150&q=80"
                        elif "lighting" in cat_lower or "lamp" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?auto=format&fit=crop&w=150&q=80"
                        elif "cabinet" in cat_lower or "wardrobe" in cat_lower:
                            img_url = "https://images.unsplash.com/photo-1595428774223-ef52624120d2?auto=format&fit=crop&w=150&q=80"
                        else:
                            img_url = "https://images.unsplash.com/photo-1513151233558-d860c5398176?auto=format&fit=crop&w=150&q=80"
                            
                    # Product link/URL validation to prevent Kaggle proxy relative URLs (Phase 3 & 4)
                    p_url = item.get("product_url") or item.get("link") or ""
                    if not (p_url.startswith("http://") or p_url.startswith("https://")):
                        p_url = "#"
                    
                    # Create card details layout with clickable product title links opening in new pages
                    card_html = f"""
                    <div style="display: flex; align-items: center; background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 12px; margin-bottom: 10px; width: 680px; color: #ffffff;">
                        <img src="{img_url}" style="width: 80px; height: 80px; border-radius: 6px; object-fit: cover; margin-right: 15px; border: 1px solid #2d3748;">
                        <div style="flex-grow: 1;">
                            <div style="font-weight: bold; font-size: 16px;">
                                <a href="{p_url}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: #38bdf8; font-weight: bold;"> {item.get('title') or item.get('name')}</a>
                            </div>
                            <div style="color: #94a3b8; font-size: 13px; margin: 4px 0;">Brand: <b>{item.get('brand')}</b> | Style: {item.get('style')}</div>
                            <div style="color: #cbd5e1; font-size: 12px; margin-bottom: 6px;"><b>Proposed Zone:</b> {item.get('proposed_location', 'Placement in empty floor coordinates.')}</div>
                            <div style="display: flex; gap: 10px; align-items: center;">
                                <span style="font-weight: bold; color: #fbbf24;">Rs. {item.get('price')}</span>
                                <span class="score-tag" style="margin-left: 10px;">Rank Score: {item.get('final_score')}</span>
                                <button onclick="window.open('{p_url}', '_blank', 'noopener,noreferrer')" style="margin-left: auto; height: 28px; padding: 0 12px; background-color: #22c55e; color: #ffffff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif; font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">🌐 View Details</button>
                            </div>
                        </div>
                    </div>
                    """
                    
                    try_btn = widgets.Button(
                        description=btn_desc,
                        button_style=btn_style,
                        layout=widgets.Layout(width="160px", height="32px", margin="4px")
                    )
                    
                    # Cache the button mapping
                    state["buttons"][p_id] = try_btn
                    
                    # Bind local callback with button self-reference
                    try_btn.on_click(lambda b, p=item: on_try_product(p, b))
                    
                    # Display card and place button side-by-side
                    display(widgets.HBox([
                        widgets.HTML(card_html),
                        try_btn
                    ], layout=widgets.Layout(align_items="center")))

                # Removed hardcoded zipping
                pass

            except Exception as e:
                print(f" Execution crashed: {str(e)}")
                raise e

    run_btn.on_click(on_run_clicked)
