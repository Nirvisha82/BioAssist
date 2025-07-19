"""
Build vector index using our ingestion pipeline and vector DB manager.
Run this from anywhere inside your project directory.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import pickle

# Add project root to path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
from src.ingestion.document_preprocessor import UniversalDocumentProcessor
from src.ingestion.vector_db_manager import VectorDBFactory
from src.utils.config_manager import ConfigManager

def build_vector_index():
    print("🔧 Building Biomedical Vector Index")
    print("=" * 60)

    start_time = time.time()

    # Load config
    config = ConfigManager()

    # Initialize processor
    processor = UniversalDocumentProcessor(config)

    # Process all files from data_source
    checkpoint_path = PROJECT_ROOT / "vector_db" / "chunks_checkpoint.pkl"
    metadata_path= PROJECT_ROOT / "vector_db" / "metadata_checkpoint.pkl"

    
    os.makedirs(checkpoint_path.parent, exist_ok=True)

    # If we already have a checkpoint, load it instead of re-processing
    if checkpoint_path.exists():
        print("🔄 Loading chunks from checkpoint...")
        with open(checkpoint_path, "rb") as f:
            all_chunks = pickle.load(f)
            
        with open(metadata_path, "rb") as g:
            metadata_list = pickle.load(g)
    else:
        print("📥 Processing documents...")
        metadata_list, all_chunks = processor.process_all_documents()
        print("💾 Saving chunks checkpoint...")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(metadata_path, "wb") as g:
            pickle.dump(metadata_list, g)

    files_processed = len(metadata_list)
    chunk_count = len(all_chunks)

    if chunk_count == 0:
        print("❌ No chunks generated or found! Aborting index build.")
        return None
    
    # Create vector DB
    print("📦 Initializing vector database...")
    vector_db = VectorDBFactory.create_vector_db(config)

    print("💾 Indexing documents...")
    try:
        vector_db.add_documents(all_chunks)
        print("⏳ Finished `add_documents` — now collecting stats...")
    except Exception as e:
        print(f"❌ Exception in add_documents: {e}")
        raise

    # Collect stats
    stats = vector_db.get_collection_stats()
    print(f"📊 Got stats: {stats}")
    elapsed = time.time() - start_time

    print("\n✅ Vector index built successfully!")
    print(f"   📁 Files processed: {files_processed}")
    print(f"   📄 Chunks created: {chunk_count}")
    print(f"   ⏱️ Time taken: {elapsed:.2f} seconds")
    print(f"   📊 Vector DB stats: {json.dumps(stats, indent=2)}")

    # Optional: write to summary report
    report_path = PROJECT_ROOT / "vector_db" / "vector_index_summary.txt"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"📁 Files processed: {files_processed}\n")
        f.write(f"📄 Chunks created: {chunk_count}\n")
        f.write(f"⏱️ Time taken: {elapsed:.2f} seconds\n")
        f.write(f"📊 Vector DB stats: {json.dumps(stats, indent=2)}\n")

    return {
        "vector_db": vector_db,
        "total_chunks": chunk_count,
        "files_processed": files_processed,
        "config": config
    }

if __name__ == "__main__":
    success = build_vector_index()
    if not success:
        sys.exit(1)
