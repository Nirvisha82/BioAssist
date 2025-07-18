#!/usr/bin/env python3
"""
Quick test to compare local vs. web retrieval in your RAGPipeline.
"""
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_manager import ConfigManager
from src.ingestion.vector_db_manager import VectorDBFactory

def main():
    config    = ConfigManager()
    vector_db = VectorDBFactory.create_vector_db(config)
    pipeline  = RAGPipeline(vector_db, config)

    query = "What are common symptoms of autoimmune hepatitis?"

    # 1) Local KB retrieval
    print("\n🔍 Local Retrieval Results:")
    local_chunks = pipeline._retrieve_from_kb(query)   # ← use _retrieve_from_kb :contentReference[oaicite:7]{index=7}
    if not local_chunks:
        print("  (no local chunks found)")
    for i, chunk in enumerate(local_chunks, 1):
        print(f"{i}. [Score: {chunk.similarity_score:.3f}] Source: {chunk.source_document}")
        print(f"   {chunk.content[:200].replace(chr(10),' ')}...\n")

    # 2) Web retrieval
    print("\n🌐 Web Retrieval Results:")
    web_chunks = pipeline._retrieve_from_web(query)    # ← use _retrieve_from_web :contentReference[oaicite:8]{index=8}
    if not web_chunks:
        print("  (no web results found)")
    for i, res in enumerate(web_chunks, 1):
        print(f"{i}. URL: {res.source}")
        print(f"   {res.snippet[:200].replace(chr(10),' ')}...\n")

if __name__ == "__main__":
    main()
