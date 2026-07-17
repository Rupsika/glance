import os
import json
import sqlite3
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

# Custom CSS styling injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
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
    
    .presets-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
        margin-bottom: 0.75rem;
    }
    
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
    
    .result-card-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
    }
    
    .rank-tag {
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #E5E7EB;
    }
    
    .score-tag {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #E0E7FF;
    }
    
    .meta-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4B5563;
    }
    
    .meta-val {
        font-size: 0.8rem;
        color: #1F2937;
    }
    
    .card-metadata {
        background-color: #F9FAFB;
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid #F3F4F6;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<p class="app-header">✨ Glance Multimodal Search</p>', unsafe_allow_html=True)
st.markdown('<p class="app-description">Intelligent Visual Retrieval Engine featuring FAISS semantic indexing, SQLite metadata extraction, and late-fusion re-ranking.</p>', unsafe_allow_html=True)

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
    
    # We check if bbox is a valid non-empty list
    if has_person and bbox and len(bbox) == 4:
        xmin, ymin, xmax, ymax = bbox
        # Draw main person bounding box in Indigo
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#4F46E5", width=4)
        
        w = xmax - xmin
        h = ymax - ymin
        
        upper_box = [xmin, ymin, xmax, min(h_img, ymin + int(0.5 * h))]
        lower_box = [xmin, ymin + int(0.4 * h), xmax, ymax]
        
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

# Make sure index and db exist
if not os.path.exists(os.path.join(INDEX_DIR, "global.index")) or not os.path.exists(os.path.join(INDEX_DIR, "metadata.db")):
    st.warning("⚠️ Search index files or SQLite database not found. Please index the dataset first.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        clip_model = st.selectbox(
            "Select CLIP Model backbone:",
            ["openai/clip-vit-base-patch32"]
        )
        batch_size = st.slider("Batch Size", 8, 128, 32)
        
        if st.button("🚀 Start Indexing Process"):
            with st.spinner("Extracting features on CPU (this takes about 2-3 minutes)... Please wait."):
                from src.indexer import main as index_main
                import sys
                old_argv = sys.argv
                sys.argv = [
                    "indexer.py",
                    "--data_dir", DATA_DIR,
                    "--output_dir", INDEX_DIR,
                    "--clip_model", clip_model,
                    "--batch_size", str(batch_size),
                    "--no_blip"
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
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Late-Fusion Weights")
st.sidebar.markdown("""
Scores are calculated via the pipeline re-ranking:
- **CLIP Similarity**: `45%`
- **Clothing Match**: `20%`
- **Color Binding**: `15%`
- **Caption Match**: `10%`
- **Scene Match**: `5%`
- **Style Match**: `5%`
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
    with st.spinner("Searching FAISS and querying SQLite metadata..."):
        try:
            results, parsed = search(
                query=st.session_state.search_query,
                index_dir=INDEX_DIR,
                data_dir=DATA_DIR,
                top_k=top_k
            )
            
            # Show Query Parser Outputs
            st.markdown("### 🔍 Query Deconstruction")
            st.markdown(f"""
            <div style="display: flex; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
                <div style="background-color: #E0F2FE; color: #0369A1; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #BAE6FD; font-size: 0.9rem;">
                    🌍 Scene: <span style="font-weight: 500; color: #1E293B;">{parsed["scene"] if parsed["scene"] else "[Any]"}</span>
                </div>
                <div style="background-color: #ECFDF5; color: #047857; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #A7F3D0; font-size: 0.9rem;">
                    👗 Clothes: <span style="font-weight: 500; color: #1E293B;">{", ".join(parsed["clothes"]) if parsed["clothes"] else "[Any]"}</span>
                </div>
                <div style="background-color: #FFF7ED; color: #C2410C; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #FFEDD5; font-size: 0.9rem;">
                    🎨 Color Bindings: <span style="font-weight: 500; color: #1E293B;">{parsed["color_bindings"] if parsed["color_bindings"] else "[Any]"}</span>
                </div>
                <div style="background-color: #F3E8FF; color: #6B21A8; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 600; border: 1px solid #E9D5FF; font-size: 0.9rem;">
                    ✨ Style: <span style="font-weight: 500; color: #1E293B;">{parsed["style"] if parsed["style"] else "[Any]"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            # Display Search Results
            st.markdown("---")
            st.markdown(f"### 📊 Top {top_k} Reranked Results")
            
            cols = st.columns(4)
            for idx, res in enumerate(results):
                col = cols[idx % 4]
                with col:
                    # Draw Bounding Box Overlay
                    overlay_img, upper_crop, lower_crop = draw_bbox_overlay(
                        res["image_abs_path"], res["bbox"], True if res["bbox"] else False
                    )
                    
                    st.image(overlay_img, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="result-card-info">
                        <span class="rank-tag">Rank {idx+1}</span>
                        <span class="score-tag">Score: {res['score']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Structured Metadata Card
                    st.markdown(f"""
                    <div class="card-metadata">
                        <div style="margin-bottom: 0.4rem;">
                            <span class="meta-label">📝 Caption:</span> <span class="meta-val">"{res['caption']}"</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
                            <div>
                                <span class="meta-label">🌍 Scene:</span> <span class="meta-val">{res['scene']}</span>
                            </div>
                            <div>
                                <span class="meta-label">✨ Style:</span> <span class="meta-val">{res['style']}</span>
                            </div>
                        </div>
                        <div style="margin-top: 0.4rem;">
                            <span class="meta-label">👕 Clothes:</span> <span class="meta-val">{", ".join(res['clothes']) if res['clothes'] else 'None'}</span>
                        </div>
                        <div style="margin-top: 0.2rem;">
                            <span class="meta-label">🎨 Colors:</span> <span class="meta-val">{json.dumps(res['colors'])}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expander for Bounding Box and Score Breakdown
                    with st.expander("🔍 Show Crops & Score Explainability"):
                        crop_col1, crop_col2 = st.columns(2)
                        with crop_col1:
                            st.image(upper_crop, caption="Upper Crop", use_container_width=True)
                        with crop_col2:
                            st.image(lower_crop, caption="Lower Crop", use_container_width=True)
                            
                        # Show Score Breakdown Progress Bars
                        st.markdown("<small>**Late-Fusion Score Weights:**</small>", unsafe_allow_html=True)
                        b = res["score_breakdown"]
                        
                        st.write(f"CLIP Image Match (45%): {b['clip']:.4f}")
                        st.progress(min(max(b["clip"], 0.0), 1.0))
                        
                        st.write(f"Caption Text Match (10%): {b['caption']:.4f}")
                        st.progress(min(max(b["caption"], 0.0), 1.0))
                        
                        st.write(f"Clothing Match (20%): {b['clothes']:.2f}")
                        st.progress(b["clothes"])
                        
                        st.write(f"Color Binding Match (15%): {b['colors']:.2f}")
                        st.progress(b["colors"])
                        
                        st.write(f"Scene Match (5%): {b['scene']:.2f}")
                        st.progress(b["scene"])
                        
                        st.write(f"Style Match (5%): {b['style']:.2f}")
                        st.progress(b["style"])
                            
        except Exception as e:
            st.error(f"Search Execution Failed: {e}")
            st.exception(e)
