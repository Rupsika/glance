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
                              help="HuggingFace model ID (e.g. openai/clip-vit-base-patch32 or patrickjohncyh/fashion-clip)")
    index_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    index_parser.add_argument("--max_images", type=int, default=None, help="Limit number of images to index")
    
    # Subparser for searching
    search_parser = subparsers.add_parser("search", help="Query the indexed dataset")
    search_parser.add_argument("--query", type=str, required=True, help="Natural language search query")
    search_parser.add_argument("--index_dir", type=str, default="index_db", help="Path to index and metadata folder")
    search_parser.add_argument("--data_dir", type=str, default="val_test2020/test", help="Path to raw image folder")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of images to retrieve")
    search_parser.add_argument("--w_global", type=float, default=0.4, help="Weight for global scene matching")
    search_parser.add_argument("--w_upper", type=float, default=0.3, help="Weight for upper body matching")
    search_parser.add_argument("--w_lower", type=float, default=0.3, help="Weight for lower body matching")
    
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
        sys.argv = sys_args
        index_main()
    elif args.command == "search":
        # Check if index exists
        if not os.path.exists(os.path.join(args.index_dir, "global.index")):
            print(f"Error: Vector index files not found in '{args.index_dir}'. Please run the 'index' command first.")
            sys.exit(1)
            
        from src.retriever import main as retriever_main
        sys.argv = [
            "retriever.py",
            "--query", args.query,
            "--index_dir", args.index_dir,
            "--data_dir", args.data_dir,
            "--top_k", str(args.top_k),
            "--w_global", str(args.w_global),
            "--w_upper", str(args.w_upper),
            "--w_lower", str(args.w_lower)
        ]
        retriever_main()
    else:
        print("Please provide a valid command: 'index' or 'search'. Run with -h for help.")

if __name__ == "__main__":
    main()
