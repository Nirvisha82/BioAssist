import sys
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import uuid
import time
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our existing classes
from src.models.data_models import DocumentChunk, DocumentMetadata, DocumentStatus, DocumentType
from src.utils.config_manager import ConfigManager
from src.ingestion.vector_db_manager import VectorDBFactory
from src.ingestion.document_preprocessor import UniversalDocumentProcessor

def create_vector_index():
    """Build complete vector index using our existing infrastructure."""
    print("🏗️ Building Vector Index with Existing Infrastructure")
    print("=" * 60)
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    start_time = time.time()
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Load config
    config = ConfigManager()
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(
        config.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2")
    )
    
    # Initialize vector database using our existing factory
    # Create vector DB
    print("📦 Initializing vector database...")
    vector_db = VectorDBFactory.create_vector_db(config)
    print("   ✅ Vector database ready")
    
    # Process all files
    all_chunks = []
    files_processed = 0
    
    processor = UniversalDocumentProcessor(config)
    metadata_list, all_chunks = processor.process_all_documents()
    files_processed = len(metadata_list)
    
    if not all_chunks:
        print("❌ No chunks created!")
        return None
    
    print(f"\n📊 Total Summary:")
    print(f"   📁 Files processed: {files_processed}")
    print(f"   📄 Chunks created: {len(all_chunks)}")
    
    # Store in vector database using our existing manager
    print(f"\n💾 Storing in vector database...")
    vector_db.add_documents(all_chunks)
    
    # Get final stats
    stats = vector_db.get_collection_stats()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Vector Index Built Successfully!")
    print(f"   ⏱️ Time taken: {elapsed_time:.1f} seconds")
    print(f"   📁 Files processed: {files_processed}")
    print(f"   📄 Total chunks: {len(all_chunks)}")
    print(f"   🔤 Embedding dimension: 384")
    print(f"   📊 Database stats: {stats}")
    
    return {
        'vector_db': vector_db,
        'embedding_model': embedding_model,
        'total_chunks': len(all_chunks),
        'files_processed': files_processed,
        'config': config
    }

def process_csv_files(embedding_model):
    """Process all CSV files."""
    print("\n📊 Processing CSV files...")
    
    data_source = PROJECT_ROOT / "data_source"
    csv_dir = data_source / "csv"
    
    if not csv_dir.exists():
        print(f"   ⚠️ CSV directory not found: {csv_dir}")
        return [], 0
    
    csv_files = list(csv_dir.glob("*.csv"))
    print(f"   Found {len(csv_files)} CSV files")
    
    all_chunks = []
    
    for csv_file in csv_files:
        try:
            chunks = process_single_csv(csv_file, embedding_model)
            all_chunks.extend(chunks)
            print(f"   ✅ {csv_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"   ❌ {csv_file.name}: {e}")
    
    return all_chunks, len(csv_files)

def process_single_csv(file_path, embedding_model):
    """Process a single CSV file into DocumentChunk objects."""
    df = pd.read_csv(file_path)
    file_name = file_path.name
    chunks = []
    
    # Create document metadata
    doc_metadata = DocumentMetadata(
        file_name=file_name,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        file_type=DocumentType.XLSX,  # Using XLSX for CSV data files
        created_at=datetime.now(),
        status=DocumentStatus.COMPLETED,
        chunk_count=0
    )
    
    # Dataset summary
    summary_content = f"""Dataset: {file_name}
    
Overview:
- Total Records: {len(df):,}
- Columns: {len(df.columns)}
- Data Fields: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}

Column Analysis:"""
    
    # Analyze columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        summary_content += f"\nNumeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}"
        for col in numeric_cols[:3]:
            stats = df[col].describe()
            summary_content += f"\n- {col}: Range {stats['min']:.1f} to {stats['max']:.1f}, Average {stats['mean']:.1f}"
    
    if len(text_cols) > 0:
        summary_content += f"\nCategorical Columns ({len(text_cols)}): {', '.join(text_cols[:5])}"
        for col in text_cols[:3]:
            top_values = df[col].value_counts().head(3)
            summary_content += f"\n- {col}: {', '.join([f'{k}({v})' for k, v in top_values.items()])}"
    
    # Generate embedding for summary
    summary_embedding = embedding_model.encode([summary_content]).tolist()[0]
    
    summary_chunk = DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=file_name,
        content=summary_content,
        metadata={
            'source_file': file_name,
            'file_type': 'csv',
            'chunk_type': 'summary',
            'record_count': len(df)
        },
        chunk_index=0,
        embedding=summary_embedding
    )
    chunks.append(summary_chunk)
    
    # Data sample chunks (first 100 rows)
    sample_df = df.head(100)
    chunk_size = 20
    
    for i in range(0, len(sample_df), chunk_size):
        chunk_df = sample_df.iloc[i:i+chunk_size]
        
        content = f"Sample data from {file_name} (rows {i+1}-{min(i+chunk_size, len(sample_df))}):\n\n"
        
        for idx, row in chunk_df.iterrows():
            record_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    if isinstance(val, float) and val.is_integer():
                        val = int(val)
                    record_parts.append(f"{col}: {val}")
            
            content += f"Record {idx+1}: {', '.join(record_parts[:8])}{'...' if len(record_parts) > 8 else ''}\n"
        
        # Generate embedding for data chunk
        data_embedding = embedding_model.encode([content]).tolist()[0]
        
        data_chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=file_name,
            content=content,
            metadata={
                'source_file': file_name,
                'file_type': 'csv',
                'chunk_type': 'data_sample',
                'row_start': i + 1,
                'row_end': min(i + chunk_size, len(sample_df))
            },
            chunk_index=len(chunks),
            embedding=data_embedding
        )
        chunks.append(data_chunk)
    
    return chunks

def process_txt_files(embedding_model):
    """Process all TXT files."""
    print("\n📝 Processing TXT files...")
    
    data_source = PROJECT_ROOT / "data_source"
    txt_dir = data_source / "txt"
    
    if not txt_dir.exists():
        print(f"   ⚠️ TXT directory not found: {txt_dir}")
        return [], 0
    
    txt_files = list(txt_dir.glob("*.txt"))
    print(f"   Found {len(txt_files)} TXT files")
    
    all_chunks = []
    
    for txt_file in txt_files:
        try:
            chunks = process_single_txt(txt_file, embedding_model)
            all_chunks.extend(chunks)
            print(f"   ✅ {txt_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"   ❌ {txt_file.name}: {e}")
    
    return all_chunks, len(txt_files)

def process_single_txt(file_path, embedding_model):
    """Process a single TXT file into DocumentChunk objects."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    file_name = file_path.name
    chunks = []
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Combine short paragraphs
    combined_chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < 1000:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                combined_chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        combined_chunks.append(current_chunk.strip())
    
    # Create DocumentChunk objects
    for i, chunk_content in enumerate(combined_chunks):
        if len(chunk_content) > 100:  # Only substantial chunks
            # Generate embedding
            chunk_embedding = embedding_model.encode([chunk_content]).tolist()[0]
            
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=file_name,
                content=chunk_content,
                metadata={
                    'source_file': file_name,
                    'file_type': 'txt',
                    'chunk_type': 'text_section',
                    'section_number': i + 1
                },
                chunk_index=i,
                embedding=chunk_embedding
            )
            chunks.append(chunk)
    
    return chunks

def process_pdf_files(embedding_model):
    """Process PDF files (placeholder implementation)."""
    print("\n📄 Processing PDF files...")
    
    data_source = PROJECT_ROOT / "data_source"
    pdf_dir = data_source / "pdf"
    
    if not pdf_dir.exists():
        print(f"   ⚠️ PDF directory not found: {pdf_dir}")
        return [], 0
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"   Found {len(pdf_files)} PDF files")
    
    chunks = []
    
    for pdf_file in pdf_files:
        content = f"PDF Document: {pdf_file.name}\n\nThis is a placeholder for PDF content. The file contains medical/research information that would be extracted using PyPDFLoader in the full implementation."
        
        # Generate embedding
        embedding = embedding_model.encode([content]).tolist()[0]
        
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=pdf_file.name,
            content=content,
            metadata={
                'source_file': pdf_file.name,
                'file_type': 'pdf',
                'chunk_type': 'placeholder'
            },
            chunk_index=0,
            embedding=embedding
        )
        chunks.append(chunk)
        print(f"   📝 {pdf_file.name}: Placeholder created")
    
    return chunks, len(pdf_files)

def process_docx_files(embedding_model):
    """Process DOCX files (placeholder implementation)."""
    print("\n📄 Processing DOCX files...")
    
    data_source = PROJECT_ROOT / "data_source"
    docx_dir = data_source / "doc"
    
    if not docx_dir.exists():
        print(f"   ⚠️ DOCX directory not found: {docx_dir}")
        return [], 0
    
    docx_files = list(docx_dir.glob("*.docx"))
    print(f"   Found {len(docx_files)} DOCX files")
    
    chunks = []
    
    for docx_file in docx_files:
        content = f"Word Document: {docx_file.name}\n\nThis is a placeholder for DOCX content. The file contains medical/research information that would be extracted using Docx2txtLoader in the full implementation."
        
        # Generate embedding
        embedding = embedding_model.encode([content]).tolist()[0]
        
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=docx_file.name,
            content=content,
            metadata={
                'source_file': docx_file.name,
                'file_type': 'docx',
                'chunk_type': 'placeholder'
            },
            chunk_index=0,
            embedding=embedding
        )
        chunks.append(chunk)
        print(f"   📝 {docx_file.name}: Placeholder created")
    
    return chunks, len(docx_files)

def test_vector_index(rag_system):
    """Test the built vector index."""
    print(f"\n🔍 Testing Vector Index")
    print("-" * 30)
    
    test_queries = [
        "COVID-19 vaccination effectiveness",
        "ICU cases and hospitalization data", 
        "autoimmune hepatitis symptoms",
        "patient health outcomes"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        try:
            # Use our existing vector DB search method
            results = rag_system['vector_db'].similarity_search(query, k=2)
            
            if results:
                for i, result in enumerate(results[:1]):  # Show top result
                    print(f"   📄 Match {i+1}: {result.source_document} (similarity: {result.similarity_score:.3f})")
                    print(f"      Preview: {result.content[:100]}...")
            else:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"   ❌ Search error: {e}")

def main():
    """Main function to build vector index."""
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    # Build vector index
    rag_system = create_vector_index()
    
    if rag_system:
        # Test the index
        test_vector_index(rag_system)
        
        print(f"\n🎉 SUCCESS!")
        print(f"Vector index built using existing infrastructure!")
        print(f"\nNext steps:")
        print(f"1. ✅ Vector index is ready")
        print(f"2. 🎯 Build RAG pipeline")
        print(f"3. 🎯 Create Streamlit interface")
        return True
    else:
        print(f"\n❌ FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)