import argparse
from src.utils.config_manager import ConfigManager
from src.ingestion.vector_db_manager import VectorDBFactory

def main():
    parser = argparse.ArgumentParser(description="Query the RAG vector index")
    parser.add_argument(
        "--query", "-q", type=str, required=True,
        help="The natural‐language query to run against the index"
    )
    parser.add_argument(
        "--k", "-k", type=int, default=5,
        help="Number of top results to retrieve (default: 5)"
    )
    args = parser.parse_args()

    # 1. Load your config (knows which backend: faiss or chroma)
    config = ConfigManager()

    # 2. Build the vector DB instance
    vector_db = VectorDBFactory.create_vector_db(config)

    # 3. Run the similarity search
    print(f"🔎 Querying for: “{args.query}” (top {args.k})\n")
    results = vector_db.similarity_search(args.query, k=args.k)

    # 4. Print out the results
    if not results:
        print("❌ No results found.")
        return

    for idx, res in enumerate(results, start=1):
        print(f"{idx}. [Score: {res.similarity_score:.3f}] Source: {res.source_document}")
        preview = res.content.replace("\n", " ")
        print(f"   {preview[:200].strip()}...\n")

if __name__ == "__main__":
    main()