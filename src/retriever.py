import os
import re
import json
import argparse
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Retriever")
    parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    parser.add_argument("--index_dir", type=str, default="index_db", help="Path to index and metadata folder")
    parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    parser.add_argument("--top_k", type=int, default=5, help="Number of images to retrieve")
    parser.add_argument("--w_global", type=float, default=0.4, help="Weight for global scene matching")
    parser.add_argument("--w_upper", type=float, default=0.3, help="Weight for upper body matching")
    parser.add_argument("--w_lower", type=float, default=0.3, help="Weight for lower body matching")
    return parser.parse_args()

def load_resources(index_dir, device):
    # Load configs
    with open(os.path.join(index_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    with open(os.path.join(index_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    clip_model_name = config["clip_model"]
    
    # Load CLIP
    print(f"Loading CLIP model '{clip_model_name}' on {device}...")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()
    
    # Load FAISS indexes
    print("Loading FAISS vector indexes...")
    global_index = faiss.read_index(os.path.join(index_dir, "global.index"))
    upper_index = faiss.read_index(os.path.join(index_dir, "upper.index"))
    lower_index = faiss.read_index(os.path.join(index_dir, "lower.index"))
    
    return model, processor, metadata, global_index, upper_index, lower_index

def parse_query(query):
    """
    Decomposes a query sentence into global context, upper body, and lower body sub-phrases.
    """
    upper_keywords = ["shirt", "t-shirt", "tshirt", "tie", "blazer", "hoodie", "jacket", "coat", "raincoat", 
                      "sweater", "blouse", "top", "suit", "cardigan", "outerwear", "button-down", "vest"]
    lower_keywords = ["pants", "jeans", "shorts", "skirt", "trousers", "leggings", "sneakers", "boots", "shoes"]
    env_keywords = ["office", "street", "park", "bench", "home", "indoor", "outdoor", "walk", "setting", 
                    "background", "garden", "nature", "outside", "inside", "interior", "office interior", "modern office"]
    
    # Split query into sub-phrases by conjunctions / prepositions
    phrases = re.split(r'\b(?:and|with|in|sitting on|inside|at|on|for a)\b', query, flags=re.IGNORECASE)
    phrases = [p.strip() for p in phrases if p.strip()]
    
    upper_phrases = []
    lower_phrases = []
    global_phrases = []
    
    for phrase in phrases:
        phrase_lower = phrase.lower()
        matched = False
        
        # Check environment keywords
        if any(ek in phrase_lower for ek in env_keywords):
            global_phrases.append(phrase)
            matched = True
            
        # Check upper body keywords
        if any(uk in phrase_lower for uk in upper_keywords):
            upper_phrases.append(phrase)
            matched = True
            
        # Check lower body keywords
        if any(lk in phrase_lower for lk in lower_keywords):
            lower_phrases.append(phrase)
            matched = True
            
        # Fallback if no direct keyword matches but has nouns or colors
        if not matched:
            # If it's a short descriptive chunk, attach it to global context
            global_phrases.append(phrase)
            
    # Compile parsed components
    parsed = {
        "global": " ".join(global_phrases) if global_phrases else query,
        "upper": " and ".join(upper_phrases) if upper_phrases else None,
        "lower": " and ".join(lower_phrases) if lower_phrases else None
    }
    
    return parsed

@torch.no_grad()
def get_text_embedding(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_features = model.get_text_features(**inputs)
    # L2 normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')

def search(query, index_dir, data_dir, top_k=5, w_global=0.4, w_upper=0.3, w_lower=0.3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load resources
    model, processor, metadata, global_index, upper_index, lower_index = load_resources(index_dir, device)
    
    # 2. Parse query
    parsed = parse_query(query)
    print("\n--- Query Decomposition ---")
    print(f"Original: '{query}'")
    print(f"Parsed Global Context: '{parsed['global']}'")
    print(f"Parsed Upper Clothing: '{parsed['upper']}'")
    print(f"Parsed Lower Clothing: '{parsed['lower']}'")
    print("---------------------------\n")
    
    # 3. Retrieve raw embeddings from FAISS
    # We reconstruct all vectors to calculate exact cosine similarities in batch
    n_records = len(metadata)
    global_vecs = global_index.reconstruct_n(0, n_records)
    upper_vecs = upper_index.reconstruct_n(0, n_records)
    lower_vecs = lower_index.reconstruct_n(0, n_records)
    
    # 4. Compute embeddings for query parts
    score = np.zeros(n_records, dtype=np.float32)
    total_weight = 0.0
    
    # Global matching
    if parsed["global"]:
        q_glob_emb = get_text_embedding(parsed["global"], model, processor, device)
        # Cosine similarity is the dot product since vectors are normalized
        sim_glob = np.dot(global_vecs, q_glob_emb.T).squeeze()
        score += w_global * sim_glob
        total_weight += w_global
        
    # Upper clothing matching
    if parsed["upper"]:
        q_upp_emb = get_text_embedding(parsed["upper"], model, processor, device)
        sim_upp = np.dot(upper_vecs, q_upp_emb.T).squeeze()
        score += w_upper * sim_upp
        total_weight += w_upper
        
    # Lower clothing matching
    if parsed["lower"]:
        q_low_emb = get_text_embedding(parsed["lower"], model, processor, device)
        sim_low = np.dot(lower_vecs, q_low_emb.T).squeeze()
        score += w_lower * sim_low
        total_weight += w_lower
        
    # Normalize score by active weights
    if total_weight > 0:
        score = score / total_weight
        
    # 5. Rank and retrieve top-k
    top_indices = np.argsort(score)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        meta = metadata[idx]
        image_abs_path = os.path.join(data_dir, meta["image_path"])
        results.append({
            "id": meta["id"],
            "image_path": meta["image_path"],
            "image_abs_path": image_abs_path,
            "has_person": meta["has_person"],
            "bbox": meta["bbox"],
            "score": float(score[idx])
        })
        
    return results, parsed

def main():
    args = parse_args()
    results, parsed = search(
        query=args.query,
        index_dir=args.index_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        w_global=args.w_global,
        w_upper=args.w_upper,
        w_lower=args.w_lower
    )
    
    print(f"Top {args.top_k} results for query '{args.query}':")
    for i, res in enumerate(results):
        print(f"{i+1}. Image: {res['image_path']} | Score: {res['score']:.4f} | BBox: {res['bbox']}")

if __name__ == "__main__":
    main()
