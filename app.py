import os
import json
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from src.retriever import search, parse_query

# Page config
st.set_page_config(
    page_title="Multimodal Fashion & Context Search Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Description
st.title("✨ Multimodal Fashion & Context Search Engine")
st.markdown("""
Retrieve specific fashion images from the database using natural language queries. 
The system breaks down search queries into **global context**, **upper-body**, and **lower-body clothing** to solve compositionality constraints zero-shot.
""")

# Bounding box visualizer
def draw_bbox_overlay(image_abs_path, bbox, has_person):
    try:
        img = Image.open(image_abs_path).convert('RGB')
    except Exception as e:
        # Return a placeholder image if loading fails
        placeholder = Image.new("RGB", (300, 450), color="#1e1e1e")
        return placeholder, placeholder, placeholder
        
    w_img, h_img = img.size
    overlay_img = img.copy()
    draw = ImageDraw.Draw(overlay_img)
    
    if has_person and bbox:
        xmin, ymin, xmax, ymax = bbox
        # Draw main person bounding box in red
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#ff3333", width=4)
        
        w = xmax - xmin
        h = ymax - ymin
        
        # Calculate Upper & Lower regions
        upper_box = [xmin, ymin, xmax, min(h_img, ymin + int(0.5 * h))]
        lower_box = [xmin, ymin + int(0.4 * h), xmax, ymax]
        
        # Upper crop in blue, Lower crop in green
        draw.rectangle(upper_box, outline="#3399ff", width=3)
        draw.rectangle(lower_box, outline="#33cc33", width=3)
        
        upper_crop = img.crop(upper_box)
        lower_crop = img.crop(lower_box)
    else:
        # Fallback bounding boxes
        draw.rectangle([0, 0, w_img, int(0.5 * h_img)], outline="#3399ff", width=3)
        draw.rectangle([0, int(0.4 * h_img), w_img, h_img], outline="#33cc33", width=3)
        upper_crop = img.crop((0, 0, w_img, int(0.5 * h_img)))
        lower_crop = img.crop((0, int(0.4 * h_img), w_img, h_img))
        
    return overlay_img, upper_crop, lower_crop

# Check for index
INDEX_DIR = "index_db"
DATA_DIR = "val_test2020/test"

if not os.path.exists(os.path.join(INDEX_DIR, "global.index")):
    st.warning("⚠️ Search index files were not found. You need to index the dataset (3,200 images) first.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        clip_model = st.selectbox(
            "Select CLIP Model backbone:",
            ["openai/clip-vit-base-patch32", "patrickjohncyh/fashion-clip"]
        )
        batch_size = st.slider("Batch Size", 8, 128, 32)
        
        if st.button("🚀 Start Indexing Process"):
            with st.spinner("Extracting features on CPU (this takes about 3-5 minutes)... Please do not close this window."):
                from src.indexer import main as index_main
                import sys
                
                # Hack sys.argv to trigger index_main with selected parameters
                old_argv = sys.argv
                sys.argv = [
                    "indexer.py",
                    "--data_dir", DATA_DIR,
                    "--output_dir", INDEX_DIR,
                    "--clip_model", clip_model,
                    "--batch_size", str(batch_size)
                ]
                
                try:
                    index_main()
                    st.success("🎉 Dataset successfully indexed! Reloading search page...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during indexing: {e}")
                finally:
                    sys.argv = old_argv
    st.stop()

# Load Index Config for Info
with open(os.path.join(INDEX_DIR, "config.json"), "r") as f:
    idx_config = json.load(f)

# Sidebar Configuration
st.sidebar.title("⚙️ Search Controls")
st.sidebar.markdown(f"**Indexed Model:** `{idx_config['clip_model']}`")
st.sidebar.markdown(f"**Total Database Size:** `{idx_config['total_records']} images`")

st.sidebar.markdown("### 🎛️ Fusion Weight Configuration")
w_global = st.sidebar.slider("Global Context Weight ($w_g$)", 0.0, 1.0, 0.4, 0.1)
w_upper = st.sidebar.slider("Upper Body Clothing Weight ($w_u$)", 0.0, 1.0, 0.3, 0.1)
w_lower = st.sidebar.slider("Lower Body Clothing Weight ($w_l$)", 0.0, 1.0, 0.3, 0.1)

# Normalize weights
sum_w = w_global + w_upper + w_lower
if sum_w > 0:
    w_glob_norm = w_global / sum_w
    w_upp_norm = w_upper / sum_w
    w_low_norm = w_lower / sum_w
else:
    w_glob_norm, w_upp_norm, w_low_norm = 0.4, 0.3, 0.3

st.sidebar.info(f"""
Normalized Weights:
- Global Context: `{w_glob_norm:.2f}`
- Upper Body: `{w_upp_norm:.2f}`
- Lower Body: `{w_low_norm:.2f}`
""")

top_k = st.sidebar.slider("Number of results (Top-K)", 1, 24, 8)

# Preset Query Selector
st.markdown("### 💡 Try Evaluation Queries")
eval_queries = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting."
]

preset_query = st.selectbox("Select an evaluation query to populate search:", ["Custom Search..."] + eval_queries)

# Main query input
if preset_query != "Custom Search...":
    search_query = st.text_input("Enter natural language query:", value=preset_query)
else:
    search_query = st.text_input("Enter natural language query:", value="A red tie and a white shirt in a formal setting.")

# Execute Search
if search_query:
    with st.spinner("Searching and ranking indexed representations..."):
        try:
            results, parsed = search(
                query=search_query,
                index_dir=INDEX_DIR,
                data_dir=DATA_DIR,
                top_k=top_k,
                w_global=w_glob_norm,
                w_upper=w_upp_norm,
                w_lower=w_low_norm
            )
            
            # Show Query Parser Outputs
            st.markdown("### 🔍 Query Deconstruction")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌍 Global Scene Context", parsed["global"])
            with col2:
                st.metric("👕 Upper Clothing Target", parsed["upper"] if parsed["upper"] else "[None detected]")
            with col3:
                st.metric("👖 Lower Clothing Target", parsed["lower"] if parsed["lower"] else "[None detected]")
                
            # Display Search Results
            st.markdown(f"### 📊 Top {top_k} Retrieval Results")
            
            cols = st.columns(4)
            for idx, res in enumerate(results):
                col = cols[idx % 4]
                with col:
                    # Draw Bounding Box Overlay on the fly
                    overlay_img, upper_crop, lower_crop = draw_bbox_overlay(
                        res["image_abs_path"], res["bbox"], res["has_person"]
                    )
                    
                    st.image(overlay_img, caption=f"Rank {idx+1} | Score: {res['score']:.4f}", use_container_width=True)
                    
                    # Show Bounding Box crops in expandable section
                    with st.expander("🔍 Show Spatial Crops"):
                        crop_col1, crop_col2 = st.columns(2)
                        with crop_col1:
                            st.image(upper_crop, caption="Upper Crop", use_container_width=True)
                        with crop_col2:
                            st.image(lower_crop, caption="Lower Crop", use_container_width=True)
                            
        except Exception as e:
            st.error(f"Search Execution Failed: {e}")
            st.exception(e)
