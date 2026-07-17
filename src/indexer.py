import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import faiss

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Indexer")
    parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    parser.add_argument("--output_dir", type=str, default="index_db", help="Path to save index and metadata")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", 
                        help="HuggingFace model ID (e.g. openai/clip-vit-base-patch32 or patrickjohncyh/fashion-clip)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images to index")
    return parser.parse_args()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_models(clip_model_name, device):
    print(f"Loading YOLOv8 detector...")
    yolo_model = YOLO("yolov8n.pt")
    
    print(f"Loading CLIP model '{clip_model_name}'...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    clip_model.eval()
    return yolo_model, clip_model, clip_processor

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
    
    # Identify the largest person in the scene
    for box in boxes:
        if int(box.cls[0]) == 0:  # Class 0 is 'person' in COCO
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
        
        # Upper body: top 50% of the bounding box
        upper_box = (xmin, ymin, xmax, min(h_img, ymin + int(0.5 * h)))
        # Lower body: bottom 60% of the bounding box (with some overlap)
        lower_box = (xmin, max(0, ymin + int(0.4 * h)), xmax, ymax)
        
        upper_crop = img.crop(upper_box)
        lower_crop = img.crop(lower_box)
        has_person = True
        bbox = [xmin, ymin, xmax, ymax]
    else:
        # Fallback spatial split if no person is detected
        upper_crop = img.crop((0, 0, w_img, int(0.5 * h_img)))
        lower_crop = img.crop((0, int(0.4 * h_img), w_img, h_img))
        has_person = False
        bbox = None
        
    return img, upper_crop, lower_crop, has_person, bbox

@torch.no_grad()
def extract_embeddings_batch(images, model, processor, device):
    """
    Computes normalized embeddings for a batch of PIL images.
    """
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    image_features = model.get_image_features(**inputs)
    # L2 normalization for Cosine Similarity search
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load models
    yolo_model, clip_model, clip_processor = load_models(args.clip_model, device)
    
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
        
    # 3. Process images and extract crops
    metadata = []
    global_images = []
    upper_images = []
    lower_images = []
    
    # We will accumulate crops and extract embeddings in batches
    batch_img_paths = []
    batch_has_person = []
    batch_bboxes = []
    
    global_features_list = []
    upper_features_list = []
    lower_features_list = []
    
    print("Processing images (Person detection & spatial cropping)...")
    for i, path in enumerate(tqdm(image_paths)):
        img, upper_crop, lower_crop, has_person, bbox = extract_crops(path, yolo_model)
        if img is None:
            continue
            
        global_images.append(img)
        upper_images.append(upper_crop)
        lower_images.append(lower_crop)
        
        # Relativize path for database portability
        rel_path = os.path.relpath(path, start=args.data_dir)
        batch_img_paths.append(rel_path)
        batch_has_person.append(has_person)
        batch_bboxes.append(bbox)
        
        # Trigger batch inference when batch size is reached
        if len(global_images) >= args.batch_size or i == len(image_paths) - 1:
            # Global
            g_emb = extract_embeddings_batch(global_images, clip_model, clip_processor, device)
            global_features_list.append(g_emb)
            
            # Upper
            u_emb = extract_embeddings_batch(upper_images, clip_model, clip_processor, device)
            upper_features_list.append(u_emb)
            
            # Lower
            l_emb = extract_embeddings_batch(lower_images, clip_model, clip_processor, device)
            lower_features_list.append(l_emb)
            
            # Metadata update
            for idx in range(len(global_images)):
                metadata.append({
                    "id": len(metadata),
                    "image_path": batch_img_paths[idx],
                    "has_person": batch_has_person[idx],
                    "bbox": batch_bboxes[idx]
                })
                
            # Clear batches
            global_images = []
            upper_images = []
            lower_images = []
            batch_img_paths = []
            batch_has_person = []
            batch_bboxes = []
            
    # Concatenate features
    global_features = np.concatenate(global_features_list, axis=0).astype('float32')
    upper_features = np.concatenate(upper_features_list, axis=0).astype('float32')
    lower_features = np.concatenate(lower_features_list, axis=0).astype('float32')
    
    embedding_dim = global_features.shape[1]
    print(f"Extracted feature dimensions: {embedding_dim}")
    
    # 4. Build and save FAISS Indexes
    print("Building FAISS vector indexes...")
    # IndexFlatIP uses Inner Product (Cosine Similarity because vectors are normalized)
    global_index = faiss.IndexFlatIP(embedding_dim)
    upper_index = faiss.IndexFlatIP(embedding_dim)
    lower_index = faiss.IndexFlatIP(embedding_dim)
    
    global_index.add(global_features)
    upper_index.add(upper_features)
    lower_index.add(lower_features)
    
    # Save indexes
    faiss.write_index(global_index, os.path.join(args.output_dir, "global.index"))
    faiss.write_index(upper_index, os.path.join(args.output_dir, "upper.index"))
    faiss.write_index(lower_index, os.path.join(args.output_dir, "lower.index"))
    
    # Save metadata database and configs
    config = {
        "clip_model": args.clip_model,
        "embedding_dim": embedding_dim,
        "total_records": len(metadata)
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Indexing completed successfully! Indexes and metadata saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
