import os
import json
import sqlite3
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import faiss

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Indexer (SQLite + FAISS)")
    parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    parser.add_argument("--output_dir", type=str, default="index_db", help="Path to save index and metadata")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", 
                        help="HuggingFace CLIP model ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images to index")
    parser.add_argument("--no_blip", action="store_true", default=True, 
                        help="Skip heavy BLIP captioner on CPU and generate high-quality templated captions instead")
    return parser.parse_args()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_models(clip_model_name, use_blip, device):
    print(f"Loading YOLOv8 detector...")
    yolo_model = YOLO("yolov8n.pt")
    
    print(f"Loading CLIP model '{clip_model_name}'...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    
    blip_model = None
    blip_processor = None
    if use_blip:
        print("Loading BLIP captioning model...")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()
        
    return yolo_model, clip_model, clip_processor, blip_model, blip_processor

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            caption TEXT,
            scene TEXT,
            style TEXT,
            clothes TEXT,  -- JSON list
            colors TEXT,   -- JSON dict
            bbox TEXT      -- JSON list [xmin, ymin, xmax, ymax]
        )
    """)
    conn.commit()
    return conn

def extract_crops(image_path, yolo_model):
    """
    Loads an image, detects the primary person (largest bounding box), 
    and crops the upper and lower body parts. Falls back to horizontal splits if no person is found.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"\n[Warning] Could not open image {image_path}: {e}")
        return None, None, None, False, None

    w_img, h_img = img.size
    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes
    
    person_box = None
    max_area = 0
    
    for box in boxes:
        if int(box.cls[0]) == 0:  # Class 0 is 'person'
            coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            if area > max_area:
                max_area = area
                person_box = coords
                
    if person_box is not None:
        xmin, ymin, xmax, ymax = person_box
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(w_img, int(xmax))
        ymax = min(h_img, int(ymax))
        
        w = xmax - xmin
        h = ymax - ymin
        
        upper_box = (xmin, ymin, xmax, min(h_img, ymin + int(0.5 * h)))
        lower_box = (xmin, max(0, ymin + int(0.4 * h)), xmax, ymax)
        
        upper_crop = img.crop(upper_box)
        lower_crop = img.crop(lower_box)
        has_person = True
        bbox = [xmin, ymin, xmax, ymax]
    else:
        upper_crop = img.crop((0, 0, w_img, int(0.5 * h_img)))
        lower_crop = img.crop((0, int(0.4 * h_img), w_img, h_img))
        has_person = False
        bbox = None
        
    return img, upper_crop, lower_crop, has_person, bbox

def classify_zero_shot(image, labels, model, processor, device):
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return labels[probs.argmax()]

@torch.no_grad()
def extract_embeddings_batch(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def generate_blip_caption(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, "metadata.db")
    
    # Check if database already exists; delete to build a clean index
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    # 1. Load models
    yolo_model, clip_model, clip_processor, blip_model, blip_processor = load_models(
        args.clip_model, not args.no_blip, device
    )
    
    # 2. Collect image paths
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, f))
                
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
                
    print(f"Found {len(image_paths)} images to index.")
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return
        
    # Categories for Zero-Shot classification
    scenes = ["office", "park", "street", "home", "indoor", "outdoor", "cafe", "mall"]
    styles = ["formal", "casual", "streetwear", "business"]
    
    upper_cats = ["shirt", "blazer", "hoodie", "jacket", "coat", "sweater", "tie", "dress", "top", "none"]
    lower_cats = ["pants", "jeans", "shorts", "skirt", "shoes", "boots", "none"]
    
    colors_list = ["red", "blue", "yellow", "black", "white", "brown", "green", "grey", "orange", "none"]

    global_images = []
    batch_img_paths = []
    batch_metadata = []
    global_features_list = []
    
    print("Extracting attributes and generating embeddings...")
    for i, path in enumerate(tqdm(image_paths)):
        img, upper_crop, lower_crop, has_person, bbox = extract_crops(path, yolo_model)
        if img is None:
            continue
            
        # 1. Zero-shot classifications
        scene = classify_zero_shot(img, scenes, clip_model, clip_processor, device)
        style = classify_zero_shot(img, styles, clip_model, clip_processor, device)
        
        upper_cat = classify_zero_shot(upper_crop, upper_cats, clip_model, clip_processor, device)
        upper_color = classify_zero_shot(upper_crop, colors_list, clip_model, clip_processor, device)
        
        lower_cat = classify_zero_shot(lower_crop, lower_cats, clip_model, clip_processor, device)
        lower_color = classify_zero_shot(lower_crop, colors_list, clip_model, clip_processor, device)
        
        # Build clothes & color structures
        clothes = []
        colors = {}
        
        if upper_cat != "none":
            clothes.append(upper_cat)
            if upper_color != "none":
                colors[upper_cat] = upper_color
                
        if lower_cat != "none" and lower_cat not in clothes:
            clothes.append(lower_cat)
            if lower_color != "none":
                colors[lower_cat] = lower_color
                
        # 2. Caption generation
        if not args.no_blip and blip_model is not None:
            caption = generate_blip_caption(img, blip_model, blip_processor, device)
        else:
            # High-quality templated fallback caption
            items_str = []
            if upper_cat != "none":
                items_str.append(f"a {upper_color if upper_color != 'none' else ''} {upper_cat}".replace("  ", " ").strip())
            if lower_cat != "none":
                items_str.append(f"a {lower_color if lower_color != 'none' else ''} {lower_cat}".replace("  ", " ").strip())
            
            if items_str:
                items_part = " wearing " + " and ".join(items_str)
            else:
                items_part = " wearing clothes"
                
            caption = f"A person{items_part} in a {style} style setting inside a {scene}."
            
        rel_path = os.path.relpath(path, start=args.data_dir)
        
        global_images.append(img)
        batch_img_paths.append(rel_path)
        batch_metadata.append({
            "caption": caption,
            "scene": scene,
            "style": style,
            "clothes": clothes,
            "colors": colors,
            "bbox": bbox
        })
        
        # Batch embedding extraction
        if len(global_images) >= args.batch_size:
            g_emb = extract_embeddings_batch(global_images, clip_model, clip_processor, device)
            global_features_list.append(g_emb)
            
            # Save metadata to SQLite
            for idx in range(len(global_images)):
                img_id = (len(global_features_list) - 1) * args.batch_size + idx
                meta = batch_metadata[idx]
                cursor.execute("""
                    INSERT INTO metadata (id, image_path, caption, scene, style, clothes, colors, bbox)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (img_id, batch_img_paths[idx], meta["caption"], meta["scene"], meta["style"],
                      json.dumps(meta["clothes"]), json.dumps(meta["colors"]), json.dumps(meta["bbox"])))
                
            # Reset batch lists
            global_images = []
            batch_img_paths = []
            batch_metadata = []

    # Flush remaining batch
    if len(global_images) > 0:
        g_emb = extract_embeddings_batch(global_images, clip_model, clip_processor, device)
        global_features_list.append(g_emb)
        
        for idx in range(len(global_images)):
            # Calculate correct sequential ID
            prev_records = sum(len(x) for x in global_features_list[:-1])
            img_id = prev_records + idx
            meta = batch_metadata[idx]
            cursor.execute("""
                INSERT INTO metadata (id, image_path, caption, scene, style, clothes, colors, bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (img_id, batch_img_paths[idx], meta["caption"], meta["scene"], meta["style"],
                  json.dumps(meta["clothes"]), json.dumps(meta["colors"]), json.dumps(meta["bbox"])))
            
    conn.commit()
    conn.close()
    
    # 3. Concatenate Visual Embeddings and save FAISS Index
    global_features = np.concatenate(global_features_list, axis=0).astype('float32')
    embedding_dim = global_features.shape[1]
    
    print(f"Total features extracted: {len(global_features)} with dimension {embedding_dim}")
    print("Building FAISS vector index...")
    global_index = faiss.IndexFlatIP(embedding_dim)
    global_index.add(global_features)
    
    faiss.write_index(global_index, os.path.join(args.output_dir, "global.index"))
    
    # Save config
    config = {
        "clip_model": args.clip_model,
        "embedding_dim": embedding_dim,
        "total_records": len(global_features)
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Indexing completed successfully! SQLite database and FAISS index saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
