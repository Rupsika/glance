import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Retrieval System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subparser for indexing
    index_parser = subparsers.add_parser("index", help="Index the raw image dataset")
    index_parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    index_parser.add_argument("--output_dir", type=str, default="index_db", help="Path to save index and metadata")
    index_parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", 
                              help="HuggingFace model ID")
    index_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    index_parser.add_argument("--max_images", type=int, default=None, help="Limit number of images to index")
    index_parser.add_argument("--no_blip", action="store_true", default=True,
                              help="Use template fallback instead of heavy BLIP model")
    
    # Subparser for searching
    search_parser = subparsers.add_parser("search", help="Query the indexed dataset")
    search_parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    search_parser.add_argument("--index_dir", type=str, default="index_db", help="Path to index and database folder")
    search_parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of images to retrieve")
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.command == "index":
        from src.indexer import main as index_main
        sys_args = [
            "indexer.py",
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--clip_model", args.clip_model,
            "--batch_size", str(args.batch_size)
        ]
        if args.max_images is not None:
            sys_args.extend(["--max_images", str(args.max_images)])
        if args.no_blip:
            sys_args.append("--no_blip")
        else:
            # If user explicitly unsets no_blip by passing it, we don't append it
            pass
        sys.argv = sys_args
        index_main()
    elif args.command == "search":
        # Check if index exists
        if not os.path.exists(os.path.join(args.index_dir, "global.index")):
            print(f"Error: FAISS vector index files not found in '{args.index_dir}'. Please run the 'index' command first.")
            sys.exit(1)
            
        from src.retriever import main as retriever_main
        sys.argv = [
            "retriever.py",
            "--query", args.query,
            "--index_dir", args.index_dir,
            "--data_dir", args.data_dir,
            "--top_k", str(args.top_k)
        ]
        retriever_main()
    else:
        print("Please provide a valid command: 'index' or 'search'. Run with -h for help.")

if __name__ == "__main__":
    main()
