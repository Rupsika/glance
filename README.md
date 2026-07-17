# Multimodal Fashion & Context Retrieval System

This repository implements the **Spatial-Semantic Hybrid Retrieval System** developed for the Glance ML Internship Assignment. 

The system retrieves specific fashion images based on natural language descriptions by understanding both global context ("where" they are / "vibe") and fine-grained clothing attributes ("what" they are wearing), while explicitly solving vision-language compositionality issues.

---

## 🚀 Key Features
- **Spatial Object Grounding**: Detects human subjects via YOLOv8 and segments them into Upper Body and Lower Body crop zones.
- **Compositional Search**: Isolates clothing attributes (e.g. matching "red tie + white shirt" to the upper crop, and "blue pants" to the lower crop) to eliminate color/item binding leaks.
- **Multimodal Vector Indexing**: Encodes representations using CLIP (`openai/clip-vit-base-patch32` or `Fashion-CLIP`) and stores them in FAISS vector indexes.
- **Weighted Semantic Fusion**: Combines search scores from multiple spatial regions dynamically depending on the user query.
- **Interactive Web App**: A visual dashboard built with Streamlit that overlays color-coded bounding boxes and lets you view individual body crops and tune weights.

---

## 📂 Codebase Structure
```
glance/
├── src/
│   ├── indexer.py       # YOLO detection, cropping, and CLIP indexing pipeline
│   └── retriever.py     # Query parser, vector search, and similarity fusion
├── app.py               # Streamlit interactive search engine UI
├── main.py              # CLI entrypoint to run indexing and queries
├── generate_report.py   # Script to compile report.md to PDF
├── report.md            # Written report covering methodology & analysis
├── submission_report.pdf# Formatted PDF report for submission
└── val_test2020/        # Raw dataset folder (3,200 Fashionpedia images)
```

---

## 🛠️ Setup Instructions

Ensure Python 3.8+ is installed on your system. 

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision ultralytics transformers sentence-transformers faiss-cpu reportlab streamlit pillow numpy pandas scikit-learn
   ```

2. **Verify Dataset Path**:
   Ensure your images are located in the `val_test2020/test` folder.

---

## 🖥️ How to Run

### 1. Feature Indexing (Part A: The Indexer)
Run the offline indexer to extract spatial visual features for the images. On a CPU, indexing 1,000 images takes ~3-4 minutes.
```bash
# Index the first 1,000 images (fulfills the assignment size requirements)
python main.py index --data_dir val_test2020/test --output_dir index_db --max_images 1000

# To index all 3,200 images
python main.py index --data_dir val_test2020/test --output_dir index_db
```
*Note: This downloads `yolov8n.pt` and `openai/clip-vit-base-patch32` on the first run.*

### 2. Search Queries (Part B: The Retriever)

#### Command-Line Interface (CLI):
You can execute searches directly from your terminal:
```bash
python main.py search --query "A red tie and a white shirt in a formal setting." --top_k 5
```

#### Streamlit Web App (Interactive Dashboard):
To launch the interactive visual search engine with real-time bounding box overlays:
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

For details on our design trade-offs, scalability analysis, and future work, please read the [report.md](report.md) or open the compiled [submission_report.pdf](submission_report.pdf).
