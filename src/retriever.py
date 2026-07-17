import os
import re
import json
import sqlite3
import argparse
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Retriever (SQLite + FAISS)")
    parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    parser.add_argument("--index_dir", type=str, default="index_db", help="Path to index and database folder")
    parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    parser.add_argument("--top_k", type=int, default=5, help="Number of images to retrieve")
    return parser.parse_args()

def load_resources(index_dir, device):
    # Load configs
    with open(os.path.join(index_dir, "config.json"), "r") as f:
        config = json.load(f)
        
    clip_model_name = config["clip_model"]
    
    # Load CLIP
    print(f"Loading CLIP model '{clip_model_name}' on {device}...")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()
    
    # Load FAISS index
    print("Loading FAISS vector index...")
    global_index = faiss.read_index(os.path.join(index_dir, "global.index"))
    
    return model, processor, global_index

def parse_query(query):
    """
    Decomposes a query sentence into structured components (scene, style, clothes list, color bindings).
    """
    query_lower = query.lower()
    
    colors = ["red", "blue", "yellow", "black", "white", "brown", "green", "grey", "orange"]
    query_colors = [c for c in colors if c in query_lower]
    
    clothes = ["shirt", "t-shirt", "tshirt", "blazer", "hoodie", "jacket", "coat", "sweater", "tie", "dress", 
               "pants", "jeans", "shorts", "skirt", "shoes", "boots", "bag", "hat", "attire", "suit", "raincoat"]
    
    query_clothes = []
    for c in clothes:
        if c in query_lower:
            if c in ["t-shirt", "tshirt"]:
                query_clothes.append("shirt")
            elif c == "raincoat":
                query_clothes.append("coat")
            else:
                query_clothes.append(c)
                
    # Extract color bindings (e.g., "red tie" -> tie: red)
    color_bindings = {}
    clauses = re.split(r'\b(?:and|with|in|sitting on|inside|at|on|for a)\b', query, flags=re.IGNORECASE)
    for clause in clauses:
        clause_lower = clause.strip().lower()
        clause_color = [c for c in colors if c in clause_lower]
        clause_item = [c for c in query_clothes if c in clause_lower]
        if clause_color and clause_item:
            color_bindings[clause_item[0]] = clause_color[0]
            
    # Extract scene
    scenes = ["office", "park", "street", "home", "indoor", "outdoor", "cafe", "mall"]
    query_scene = None
    for s in scenes:
        if s in query_lower:
            query_scene = s
            break
            
    # Extract style
    styles = ["formal", "casual", "streetwear", "business"]
    query_style = None
    for st in styles:
        if st in query_lower:
            query_style = st
            break
            
    return {
        "colors": query_colors,
        "clothes": list(set(query_clothes)),
        "color_bindings": color_bindings,
        "scene": query_scene,
        "style": query_style
    }

@torch.no_grad()
def get_text_embedding(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')

def search(query, index_dir, data_dir, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load resources
    model, processor, global_index = load_resources(index_dir, device)
    
    # 2. Parse query
    parsed = parse_query(query)
    print("\n--- Query Deconstruction ---")
    print(f"Original: '{query}'")
    print(f"Extracted Scenes: '{parsed['scene']}'")
    print(f"Extracted Styles: '{parsed['style']}'")
    print(f"Extracted Clothes: {parsed['clothes']}")
    print(f"Color Bindings: {parsed['color_bindings']}")
    print("----------------------------\n")
    
    # 3. Retrieve Top-100 candidates from FAISS
    q_emb = get_text_embedding(query, model, processor, device)
    D, I = global_index.search(q_emb, min(100, global_index.ntotal))
    candidate_ids = I[0].tolist()
    clip_scores = D[0].tolist()
    
    # Connect to SQLite database to fetch metadata
    db_path = os.path.join(index_dir, "metadata.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database metadata.db not found in {index_dir}")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    placeholders = ",".join("?" for _ in candidate_ids)
    cursor.execute(f"""
        SELECT id, image_path, caption, scene, style, clothes, colors, bbox 
        FROM metadata 
        WHERE id IN ({placeholders})
    """, candidate_ids)
    
    rows = cursor.fetchall()
    conn.close()
    
    meta_dict = {}
    for row in rows:
        meta_dict[row[0]] = {
            "image_path": row[1],
            "caption": row[2],
            "scene": row[3],
            "style": row[4],
            "clothes": json.loads(row[5]),
            "colors": json.loads(row[6]),
            "bbox": json.loads(row[7]) if row[7] else None
        }
        
    # 4. Sentence similarity of query against candidate captions in batch
    captions = [meta_dict[cid]["caption"] for cid in candidate_ids]
    cap_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        cap_features = model.get_text_features(**cap_inputs)
        cap_features = cap_features / cap_features.norm(dim=-1, keepdim=True)
    
    q_tensor = torch.tensor(q_emb).to(device)
    cap_sims = torch.matmul(q_tensor, cap_features.T).squeeze().cpu().numpy()
    
    # Handle single element edge-case
    if len(candidate_ids) == 1:
        cap_sims = np.array([cap_sims])
        
    # 5. Hybrid re-ranking
    ranked_results = []
    for i, cid in enumerate(candidate_ids):
        cand_meta = meta_dict[cid]
        
        # Clip Score
        score_clip = clip_scores[i]
        
        # Caption similarity
        score_caption = float(cap_sims[i])
        
        # Clothing category Jaccard match
        q_clothes = parsed["clothes"]
        cand_clothes = cand_meta["clothes"]
        if q_clothes:
            match_count = sum(1 for c in q_clothes if c in cand_clothes or (c == "attire" and len(cand_clothes) > 0))
            score_clothes = match_count / len(q_clothes)
        else:
            score_clothes = 1.0
            
        # Color binding match
        color_bindings = parsed["color_bindings"]
        cand_colors = cand_meta["colors"]
        if color_bindings:
            match_count = 0
            for item, color in color_bindings.items():
                if item in cand_colors and cand_colors[item] == color:
                    match_count += 1
                elif item == "attire" and any(c == color for c in cand_colors.values()):
                    match_count += 1
            score_colors = match_count / len(color_bindings)
        else:
            score_colors = 1.0
            
        # Scene match
        q_scene = parsed["scene"]
        cand_scene = cand_meta["scene"]
        if q_scene:
            score_scene = 1.0 if q_scene == cand_scene or (q_scene in ["indoor", "outdoor"] and cand_scene in ["indoor", "outdoor"]) else 0.0
        else:
            score_scene = 1.0
            
        # Style match
        q_style = parsed["style"]
        cand_style = cand_meta["style"]
        if q_style:
            score_style = 1.0 if q_style == cand_style else 0.0
        else:
            score_style = 1.0
            
        # Scoring function:
        # 0.45 * CLIP + 0.10 * Caption + 0.20 * Clothes + 0.15 * Colors + 0.05 * Scene + 0.05 * Style
        final_score = (
            0.45 * score_clip +
            0.10 * score_caption +
            0.20 * score_clothes +
            0.15 * score_colors +
            0.05 * score_scene +
            0.05 * score_style
        )
        
        ranked_results.append({
            "id": cid,
            "image_path": cand_meta["image_path"],
            "image_abs_path": os.path.join(data_dir, cand_meta["image_path"]),
            "caption": cand_meta["caption"],
            "scene": cand_meta["scene"],
            "style": cand_meta["style"],
            "clothes": cand_meta["clothes"],
            "colors": cand_meta["colors"],
            "bbox": cand_meta["bbox"],
            "score": float(final_score),
            "score_breakdown": {
                "clip": float(score_clip),
                "caption": float(score_caption),
                "clothes": float(score_clothes),
                "colors": float(score_colors),
                "scene": float(score_scene),
                "style": float(score_style)
            }
        })
        
    # Sort candidates by final score descending
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k], parsed

def main():
    args = parse_args()
    results, parsed = search(
        query=args.query,
        index_dir=args.index_dir,
        data_dir=args.data_dir,
        top_k=args.top_k
    )
    
    print(f"Top {args.top_k} results for query '{args.query}':")
    for i, res in enumerate(results):
        print(f"\n{i+1}. Image: {res['image_path']} | Final Score: {res['score']:.4f}")
        print(f"   Caption: '{res['caption']}'")
        print(f"   Attributes: Scene={res['scene']} | Style={res['style']} | Clothes={res['clothes']} | Colors={res['colors']}")
        print(f"   Breakdown: {res['score_breakdown']}")

if __name__ == "__main__":
    main()
