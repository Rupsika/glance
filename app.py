import os
import json
import streamlit as st
from PIL import Image, ImageDraw
from src.retriever import search, parse_query

# Page config
st.set_page_config(
    page_title="Glance Multimodal Fashion Search Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium stylesheet injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient styling */
    .app-header {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .app-description {
        color: #6B7280;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Custom design for presets */
    .presets-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
        margin-bottom: 0.75rem;
    }
    
    /* Native button customizations */
    .stButton > button {
        border-radius: 20px !important;
        background-color: #F3F4F6 !important;
        color: #374151 !important;
        border: 1px solid #E5E7EB !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0.4rem 0.85rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton > button:hover {
        background-color: #4F46E5 !important;
        color: white !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2) !important;
        transform: translateY(-1.5px);
    }
    
    /* Search box container style */
    .stTextInput input {
        border-radius: 14px !important;
        border: 1.5px solid #D1D5DB !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.25s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    }
    
    /* Styling image crops and overlays */
    .stImage img {
        border-radius: 16px !important;
        border: 1px solid #F3F4F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stImage img:hover {
        transform: scale(1.025);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
    }
    
    /* Custom card tags for search results */
    .result-card-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.6rem;
        margin-bottom: 0.75rem;
    }
    
    .rank-tag {
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 0.3rem 0.7rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #E5E7EB;
    }
    
    .score-tag {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 0.3rem 0.7rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #E0E7FF;
    }
    
    /* Clean expander boxes */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #4B5563 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<p class="app-header">✨ Glance Multimodal Search</p>', unsafe_allow_html=True)
st.markdown('<p class="app-description">Intelligent visual retrieval engine for fine-grained fashion queries and scene environments.</p>', unsafe_allow_html=True)

# Bounding box visualizer
def draw_bbox_overlay(image_abs_path, bbox, has_person):
    try:
        img = Image.open(image_abs_path).convert('RGB')
    except Exception as e:
        placeholder = Image.new("RGB", (300, 450), color="#F3F4F6")
        return placeholder, placeholder, placeholder
        
    w_img, h_img = img.size
    overlay_img = img.copy()
    draw = ImageDraw.Draw(overlay_img)
    
    if has_person and bbox:
        xmin, ymin, xmax, ymax = bbox
        # Draw main person bounding box in Indigo
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#4F46E5", width=4)
        
        w = xmax - xmin
        h = ymax - ymin
        
        # Calculate Upper & Lower regions
        upper_box = [xmin, ymin, xmax, min(h_img, ymin + int(0.5 * h))]
        lower_box = [xmin, ymin + int(0.4 * h), xmax, ymax]
        
        # Upper crop in Cyan, Lower crop in Coral
        draw.rectangle(upper_box, outline="#06B6D4", width=3)
        draw.rectangle(lower_box, outline="#F43F5E", width=3)
        
        upper_crop = img.crop(upper_box)
        lower_crop = img.crop(lower_box)
    else:
        # Fallback bounding boxes
        draw.rectangle([0, 0, w_img, int(0.5 * h_img)], outline="#06B6D4", width=3)
        draw.rectangle([0, int(0.4 * h_img), w_img, h_img], outline="#F43F5E", width=3)
        upper_crop = img.crop((0, 0, w_img, int(0.5 * h_img)))
        lower_crop = img.crop((0, int(0.4 * h_img), w_img, h_img))
        
    return overlay_img, upper_crop, lower_crop

INDEX_DIR = "index_db"
DATA_DIR = "val_test2020/test"

# Make sure index files exist
if not os.path.exists(os.path.join(INDEX_DIR, "global.index")):
    st.warning("⚠️ Search index files were not found. Please index the dataset first.")
    col1, col2 = st.columns([1, 2])
    with col1:
        clip_model = st.selectbox(
            "Select CLIP Model backbone:",
            ["openai/clip-vit-base-patch32", "patrickjohncyh/fashion-clip"]
        )
        batch_size = st.slider("Batch Size", 8, 128, 32)
        
        if st.button("🚀 Start Indexing Process"):
            with st.spinner("Extracting features on CPU (this takes about 3-5 minutes)... Please wait."):
                from src.indexer import main as index_main
                import sys
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
                    st.success("🎉 Dataset successfully indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during indexing: {e}")
                finally:
                    sys.argv = old_argv
    st.stop()

# Load Index Config
with open(os.path.join(INDEX_DIR, "config.json"), "r") as f:
    idx_config = json.load(f)

# Sidebar Configuration
st.sidebar.title("⚙️ Search Controls")
st.sidebar.markdown(f"**Model Backbone:** `{idx_config['clip_model']}`")
st.sidebar.markdown(f"**Database Size:** `{idx_config['total_records']} images`")

st.sidebar.markdown("### 🎛️ Fusion Weight Configuration")
st.sidebar.markdown("<small>Adjust weights to prioritize local garments vs global scenery context.</small>", unsafe_allow_html=True)
w_global = st.sidebar.slider("Global Context Weight ($w_g$)", 0.0, 1.0, 0.3, 0.1)
w_upper = st.sidebar.slider("Upper Body Weight ($w_u$)", 0.0, 1.0, 0.5, 0.1)
w_lower = st.sidebar.slider("Lower Body Weight ($w_l$)", 0.0, 1.0, 0.2, 0.1)

# Normalize weights
sum_w = w_global + w_upper + w_lower
if sum_w > 0:
    w_glob_norm = w_global / sum_w
    w_upp_norm = w_upper / sum_w
    w_low_norm = w_lower / sum_w
else:
    w_glob_norm, w_upp_norm, w_low_norm = 0.3, 0.5, 0.2

st.sidebar.info(f"""
**Active Weighting:**
* 🌍 Global Context: `{w_glob_norm:.2f}`
* 👕 Upper Clothing: `{w_upp_norm:.2f}`
* 👖 Lower Clothing: `{w_low_norm:.2f}`
""")

top_k = st.sidebar.slider("Number of results (Top-K)", 1, 24, 8)

# Preset Query state manager
if "search_query" not in st.session_state:
    st.session_state.search_query = "A red tie and a white shirt in a formal setting."

def apply_preset(query_text):
    st.session_state.search_query = query_text

# Clickable presets
st.markdown('<p class="presets-label">💡 Preset Evaluation Queries (Click to load):</p>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🌦️ Yellow Raincoat", use_container_width=True, key="p1"):
        apply_preset("A person in a bright yellow raincoat.")
        st.rerun()
with col2:
    if st.button("💼 Office Attire", use_container_width=True, key="p2"):
        apply_preset("Professional business attire inside a modern office.")
        st.rerun()
with col3:
    if st.button("🏞️ Blue Shirt on Bench", use_container_width=True, key="p3"):
        apply_preset("Someone wearing a blue shirt sitting on a park bench.")
        st.rerun()
with col4:
    if st.button("🚶 City Walk Outfit", use_container_width=True, key="p4"):
        apply_preset("Casual weekend outfit for a city walk.")
        st.rerun()
with col5:
    if st.button("👔 Red Tie & White Shirt", use_container_width=True, key="p5"):
        apply_preset("A red tie and a white shirt in a formal setting.")
        st.rerun()

# Main query input
st.write("---")
user_input = st.text_input(
    "Or type your custom natural language query below:",
    value=st.session_state.search_query,
    key="search_input_widget"
)

# Synchronize typed query back to session state
if user_input != st.session_state.search_query:
    st.session_state.search_query = user_input

# Execute Search
if st.session_state.search_query:
    with st.spinner("Searching and ranking fashion crops..."):
        try:
            results, parsed = search(
                query=st.session_state.search_query,
                index_dir=INDEX_DIR,
                data_dir=DATA_DIR,
                top_k=top_k,
                w_global=w_glob_norm,
                w_upper=w_upp_norm,
                w_lower=w_low_norm
            )
            
            # Show Query Parser Outputs with custom HTML/CSS pills
            st.markdown("### 🔍 Query Deconstruction")
            st.markdown(f"""
            <div style="display: flex; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
                <div style="background-color: #E0F2FE; color: #0369A1; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #BAE6FD; font-size: 0.9rem;">
                    🌍 Global Context: <span style="font-weight: 500; color: #1E293B;">{parsed["global"]}</span>
                </div>
                <div style="background-color: #F3E8FF; color: #6B21A8; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #E9D5FF; font-size: 0.9rem;">
                    👕 Upper clothing: <span style="font-weight: 500; color: #1E293B;">{parsed["upper"] if parsed["upper"] else "[None detected]"}</span>
                </div>
                <div style="background-color: #FCE7F3; color: #9D174D; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #FBCFE8; font-size: 0.9rem;">
                    👖 Lower clothing: <span style="font-weight: 500; color: #1E293B;">{parsed["lower"] if parsed["lower"] else "[None detected]"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            # Display Search Results
            st.markdown("---")
            st.markdown(f"### 📊 Top {top_k} Retrieval Results")
            
            cols = st.columns(4)
            for idx, res in enumerate(results):
                col = cols[idx % 4]
                with col:
                    # Draw Bounding Box Overlay
                    overlay_img, upper_crop, lower_crop = draw_bbox_overlay(
                        res["image_abs_path"], res["bbox"], res["has_person"]
                    )
                    
                    # Display overlay image
                    st.image(overlay_img, use_container_width=True)
                    
                    # Display e-commerce rank and score tag
                    st.markdown(f"""
                    <div class="result-card-info">
                        <span class="rank-tag">Rank {idx+1}</span>
                        <span class="score-tag">Similarity: {res['score']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show Bounding Box crops in expandable section
                    with st.expander("🔍 View Segmented Crops"):
                        crop_col1, crop_col2 = st.columns(2)
                        with crop_col1:
                            st.image(upper_crop, caption="Upper Crop", use_container_width=True)
                        with crop_col2:
                            st.image(lower_crop, caption="Lower Crop", use_container_width=True)
                            
        except Exception as e:
            st.error(f"Search Execution Failed: {e}")
            st.exception(e)
