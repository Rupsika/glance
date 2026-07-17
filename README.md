# Multimodal Fashion & Context Retrieval System
### SQLite + FAISS Late-Fusion Architecture

This repository implements the **SQLite + FAISS Late-Fusion Retrieval System** developed for the Glance ML Internship Assignment. 

The system retrieves specific fashion images based on natural language descriptions by mapping semantic global images, structured clothing types, color-bindings, scene environments, and overall style vibes using a hybrid re-ranking pipeline.

---

## 🚀 Key Features
- **Spatial Object Grounding**: Detects human subjects via YOLOv8 and partitions them into Upper Torso and Lower Torso crop zones.
- **Zero-Shot Attribute Extractor**: Classifies upper crops, lower crops, and full backgrounds for garments (shirt, jacket, jeans, etc.), colors (red, white, blue, etc.), scenes (office, park, street, etc.), and styles (formal, casual) zero-shot using OpenCLIP.
- **BLIP Caption Fallback**: Integrates BLIP conditional captioning with an instantaneous template caption fallback (`--no_blip`) for high-performance CPU indexing.
- **Structured Metadata Storage**: Stores global visual embeddings in FAISS (`global.index`) and structured json-metadata in a local SQLite database (`metadata.db`).
- **Hybrid Re-ranking Pipeline**: Retrieves the Top-100 candidates from FAISS and applies a late-fusion scoring weight function:
  $$Score = 0.45 \cdot CLIP + 0.10 \cdot Caption + 0.20 \cdot Clothes + 0.15 \cdot Colors + 0.05 \cdot Scene + 0.05 \cdot Style$$
- **Explainable Web Dashboard**: An interactive Streamlit app displaying metadata cards and a progress-bar explanation panel decomposing the similarity scores for each result.

---

## 🗺️ High-Level Architecture Diagram
```
                           Dataset Images
                                 │
                                 ▼
                     Offline Indexing Pipeline
                                 │
     ┌───────────────┬───────────────┬───────────────┐
     │               │               │               │
     ▼               ▼               ▼               ▼
 OpenCLIP      Fashion Parser   Scene Model      BLIP Captioner
     │               │               │               │
     │               │               │               │
     ▼               ▼               ▼               ▼
 Semantic      Clothing & Colors   Environment    Image Caption
 Embeddings
     └───────────────┬───────────────┬───────────────┘
                     │
                     ▼
             Metadata Construction
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      FAISS Index          SQLite Database
                     │
                     ▼
               Query Processing
                     │
                     ▼
              Hybrid Re-ranking
                     │
                     ▼
                  Top-K Results
```

---

## 📂 Codebase Structure
```
glance/
├── src/
│   ├── indexer.py         # YOLO torso cropping, zero-shot tags, and SQLite + FAISS indexer
│   └── retriever.py       # Query parsing, Top-100 FAISS lookup, SQLite fetch, and late-fusion re-ranker
├── app.py                 # Streamlit interactive search engine UI
├── main.py                # CLI entrypoint to run indexing and queries
├── generate_report.py     # Script to compile report.md to PDF
├── report.md              # Written system design & architecture report
├── submission_report.pdf  # Formatted PDF report for submission
└── val_test2020/          # Raw dataset folder (3,200 Fashionpedia images)
```

---

## 🛠️ Setup Instructions

Ensure Python 3.8+ is installed on your system. 

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision ultralytics transformers faiss-cpu reportlab streamlit pillow numpy pandas
   ```

2. **Verify Dataset Path**:
   Ensure your images are located in the `val_test2020/test` folder.

---

## 🖥️ How to Run

### 1. Feature Indexing (Stage 1: Offline Indexer)
Run the offline indexer to extract spatial features, classify scene/garment tags, and build the databases. On a CPU, indexing 1,000 images takes ~8-10 minutes (due to zero-shot classifiers).
```bash
# Index the first 1,000 images using the fast template caption builder
python main.py index --data_dir val_test2020/test --output_dir index_db --max_images 1000 --no_blip

# To run indexing including the heavy BLIP deep-learning captioner (recommended only on GPU)
python main.py index --data_dir val_test2020/test --output_dir index_db --max_images 1000
```
*Note: This downloads `yolov8n.pt` and `openai/clip-vit-base-patch32` on the first run.*

### 2. Search Queries (Stage 2: Online Retriever)

#### Command-Line Interface (CLI):
You can execute searches directly from your terminal:
```bash
python main.py search --query "A red tie and a white shirt in a formal setting." --top_k 5
```

#### Streamlit Web App (Interactive Dashboard):
To launch the interactive visual search engine with metadata cards and explainable late-fusion scoring:
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (usually `http://localhost:8501`) to perform searches.

---

## 📊 Evaluation Queries
The system parses and matches the following prompts successfully:
1. **Attribute Specific**: *"A person in a bright yellow raincoat."*
2. **Contextual/Place**: *"Professional business attire inside a modern office."*
3. **Complex Semantic**: *"Someone wearing a blue shirt sitting on a park bench."*
4. **Style Inference**: *"Casual weekend outfit for a city walk."*
5. **Compositional**: *"A red tie and a white shirt in a formal setting."*
