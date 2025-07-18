"""
Universal document ingestion and processing module.
Handles ANY CSV structure along with PDFs, DOCX, and TXT files.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from loguru import logger
import pandas as pd
import numpy as np

from ..models.data_models import DocumentMetadata, DocumentChunk, DocumentStatus, DocumentType
from ..utils.config_manager import ConfigManager


class UniversalCSVProcessor:
    """Universal CSV processor that adapts to any CSV structure."""
    
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.max_unique_for_categorical = 50
    
    def analyze_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze any CSV structure without assumptions."""
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'data_completeness': 0
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            # Classify column type
            if df[col].dtype in ['int64', 'float64']:
                col_info['type'] = 'numeric'
                if df[col].count() > 0:
                    col_info['stats'] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }
            else:
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio > 0.9:  # Mostly unique values
                    col_info['type'] = 'identifier'
                elif df[col].nunique() <= self.max_unique_for_categorical:
                    col_info['type'] = 'categorical'
                    col_info['top_values'] = df[col].value_counts().head(10).to_dict()
                else:
                    col_info['type'] = 'text'
                    col_info['avg_length'] = df[col].astype(str).str.len().mean()
            
            analysis['columns'][col] = col_info
        
        # Calculate overall data completeness
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.count().sum()
        analysis['data_completeness'] = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        return analysis
    
    def create_dataset_summary(self, df: pd.DataFrame, file_name: str, analysis: Dict) -> Document:
        """Create comprehensive dataset summary."""
        content = f"""Dataset: {file_name}

Summary:
- Records: {analysis['total_rows']:,}
- Fields: {analysis['total_columns']}
- Data Completeness: {analysis['data_completeness']:.1f}%

Column Information:"""
        
        # Group columns by type
        column_types = {}
        for col, info in analysis['columns'].items():
            col_type = info['type']
            if col_type not in column_types:
                column_types[col_type] = []
            column_types[col_type].append((col, info))
        
        for col_type, columns in column_types.items():
            content += f"\n\n{col_type.title()} Columns ({len(columns)}):"
            
            for col, info in columns:
                content += f"\n• {col}"
                
                if info['null_count'] > 0:
                    missing_pct = (info['null_count'] / analysis['total_rows']) * 100
                    content += f" (Missing: {missing_pct:.1f}%)"
                
                if col_type == 'numeric' and 'stats' in info:
                    stats = info['stats']
                    content += f" [Range: {stats['min']:.2f} to {stats['max']:.2f}]"
                elif col_type == 'categorical' and 'top_values' in info:
                    top_2 = list(info['top_values'].items())[:2]
                    content += f" [Top: {', '.join([f'{k}({v})' for k, v in top_2])}]"
                elif col_type == 'identifier':
                    content += f" [Unique: {info['unique_count']}]"
        
        return Document(
            page_content=content,
            metadata={
                'source_file': file_name,
                'file_type': 'csv',
                'chunk_type': 'dataset_summary',
                'record_count': analysis['total_rows'],
                'column_count': analysis['total_columns'],
                'completeness': analysis['data_completeness']
            }
        )
    
    def create_data_records(self, df: pd.DataFrame, file_name: str) -> List[Document]:
        """Create chunks from actual data records."""
        documents = []
        
        for i in range(0, len(df), self.chunk_size):
            chunk_df = df.iloc[i:i+self.chunk_size]
            end_idx = min(i + self.chunk_size, len(df))
            
            content = f"Records from {file_name} (Rows {i+1}-{end_idx}):\n\n"
            
            for idx, row in chunk_df.iterrows():
                record_parts = []
                
                for col, val in row.items():
                    if pd.notna(val):
                        # Format values appropriately
                        if isinstance(val, float):
                            if val.is_integer():
                                formatted_val = str(int(val))
                            else:
                                formatted_val = f"{val:.2f}"
                        else:
                            formatted_val = str(val)
                        
                        # Clean column name for readability
                        clean_col = col.replace('_', ' ').replace('-', ' ').title()
                        record_parts.append(f"{clean_col}: {formatted_val}")
                
                content += f"Row {idx + 1}: {', '.join(record_parts)}\n"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'source_file': file_name,
                    'file_type': 'csv',
                    'chunk_type': 'data_records',
                    'row_start': i + 1,
                    'row_end': end_idx,
                    'record_count': len(chunk_df)
                }
            ))
        
        return documents
    
    def create_statistical_analysis(self, df: pd.DataFrame, file_name: str, analysis: Dict) -> List[Document]:
        """Create statistical analysis documents."""
        documents = []
        
        # Numeric analysis
        numeric_cols = [col for col, info in analysis['columns'].items() if info['type'] == 'numeric']
        if numeric_cols:
            content = f"Numerical Analysis for {file_name}:\n\n"
            
            for col in numeric_cols:
                info = analysis['columns'][col]
                if 'stats' in info:
                    stats = info['stats']
                    content += f"{col}:\n"
                    content += f"• Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
                    content += f"• Average: {stats['mean']:.2f}\n"
                    content += f"• Standard Deviation: {stats['std']:.2f}\n"
                    
                    # Add quartiles
                    quartiles = df[col].quantile([0.25, 0.5, 0.75])
                    content += f"• Quartiles: Q1={quartiles[0.25]:.2f}, Median={quartiles[0.5]:.2f}, Q3={quartiles[0.75]:.2f}\n\n"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'source_file': file_name,
                    'file_type': 'csv',
                    'chunk_type': 'numeric_analysis',
                    'columns_analyzed': numeric_cols
                }
            ))
        
        # Categorical analysis
        categorical_cols = [col for col, info in analysis['columns'].items() if info['type'] == 'categorical']
        if categorical_cols:
            content = f"Categorical Analysis for {file_name}:\n\n"
            
            for col in categorical_cols:
                info = analysis['columns'][col]
                content += f"{col}:\n"
                content += f"• Total Categories: {info['unique_count']}\n"
                
                if 'top_values' in info:
                    content += f"• Distribution:\n"
                    total = analysis['total_rows']
                    for value, count in info['top_values'].items():
                        percentage = (count / total) * 100
                        content += f"  - {value}: {count} ({percentage:.1f}%)\n"
                content += "\n"
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'source_file': file_name,
                    'file_type': 'csv',
                    'chunk_type': 'categorical_analysis',
                    'columns_analyzed': categorical_cols
                }
            ))
        
        return documents
    
    def create_data_quality_report(self, df: pd.DataFrame, file_name: str, analysis: Dict) -> Document:
        """Create data quality and insights report."""
        content = f"Data Quality Report for {file_name}:\n\n"
        
        # Overall quality metrics
        content += f"Overall Quality:\n"
        content += f"• Data Completeness: {analysis['data_completeness']:.1f}%\n"
        content += f"• Total Data Points: {len(df) * len(df.columns):,}\n"
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            content += f"• Missing Data Points: {missing_data.sum():,}\n\n"
            content += f"Missing Data by Column:\n"
            
            for col, missing_count in missing_data[missing_data > 0].items():
                missing_pct = (missing_count / len(df)) * 100
                content += f"• {col}: {missing_count} missing ({missing_pct:.1f}%)\n"
        else:
            content += f"• No Missing Data\n"
        
        # Column type distribution
        type_counts = {}
        for info in analysis['columns'].values():
            col_type = info['type']
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        content += f"\nColumn Type Distribution:\n"
        for col_type, count in type_counts.items():
            content += f"• {col_type.title()}: {count} columns\n"
        
        # Dataset characteristics
        content += f"\nDataset Characteristics:\n"
        content += f"• Size: {len(df):,} rows × {len(df.columns)} columns\n"
        content += f"• Memory Usage: ~{df.memory_usage().sum() / 1024:.1f} KB\n"
        
        # Potential data insights
        if len(df) > 1:
            content += f"\nPotential Analysis Opportunities:\n"
            if len([col for col in analysis['columns'] if analysis['columns'][col]['type'] == 'numeric']) >= 2:
                content += f"• Correlation analysis between numeric variables\n"
            if len([col for col in analysis['columns'] if analysis['columns'][col]['type'] == 'categorical']) >= 1:
                content += f"• Distribution analysis of categorical variables\n"
            content += f"• Trend analysis across records\n"
        
        return Document(
            page_content=content,
            metadata={
                'source_file': file_name,
                'file_type': 'csv',
                'chunk_type': 'quality_report',
                'completeness': analysis['data_completeness'],
                'total_records': len(df)
            }
        )
    
    def process_csv(self, file_path: str) -> List[Document]:
        """Process any CSV file universally."""
        try:
            df = pd.read_csv(file_path)
            file_name = Path(file_path).name
            
            logger.info(f"Processing CSV: {file_name} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Analyze structure
            analysis = self.analyze_csv_structure(df)
            
            # Create comprehensive document set
            documents = []
            
            # 1. Dataset summary (always created)
            documents.append(self.create_dataset_summary(df, file_name, analysis))
            
            # 2. Data records (chunked appropriately)
            documents.extend(self.create_data_records(df, file_name))
            
            # 3. Statistical analysis (if applicable)
            documents.extend(self.create_statistical_analysis(df, file_name, analysis))
            
            # 4. Data quality report
            documents.append(self.create_data_quality_report(df, file_name, analysis))
            
            logger.info(f"Created {len(documents)} chunks from CSV {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            raise


class UniversalDocumentProcessor:
    """Universal document processor for any file type combination."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_source_dir = config.get("document_processing.data_source_dir", "./data_source")
        
        # Initialize text splitter
        self.text_splitter = TokenTextSplitter(
            chunk_size=config.get("embeddings.chunk_size", 500),
            chunk_overlap=config.get("embeddings.chunk_overlap", 100),
            length_function=len
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Initialize universal CSV processor
        self.csv_processor = UniversalCSVProcessor(
            chunk_size=config.get("csv.chunk_size", 10)
        )
        
        logger.info(f"Initialized Universal Document Processor")
        logger.info(f"Data source directory: {self.data_source_dir}")
        logger.info(f"Supports: PDF, DOCX, TXT, CSV (any structure)")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect duplicates."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_document_type(self, file_path: str) -> DocumentType:
        """Determine document type from file extension."""
        extension = Path(file_path).suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,

            '.txt': DocumentType.TXT,
            '.csv': DocumentType.XLSX,  # Using XLSX enum for data files
        }
        return type_mapping.get(extension, DocumentType.TXT)
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load any document type universally."""
        try:
            file_extension = Path(file_path).suffix.lower()
            file_name = Path(file_path).name
            file_path=file_path.replace("\\","/")

            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_extension == '.docx':
                try:
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                except Exception as e:
                    raise ValueError(f"DOCX parsing failed: {e}")
            elif file_extension=='.txt':
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                except Exception as e:
                    try:
                        loader = TextLoader(file_path, encoding='utf-16') #works
                        documents = loader.load()
                    except UnicodeDecodeError:
                        loader = TextLoader(file_path, encoding='cp1252')  # fallback for weird Windows text
                        documents = loader.load()
                    print(e)
                    
            elif file_extension == '.csv':
                # Use universal CSV processor
                return self.csv_processor.process_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Add metadata for non-CSV files
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_name,
                    'file_type': file_extension[1:],
                    'file_path': file_path
                })
            
            logger.info(f"Loaded {len(documents)} sections from {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents appropriately by type."""
        try:
            chunks = []
            for doc in documents:
                # CSV documents are already optimally chunked
                if doc.metadata.get('file_type') == 'csv':
                    chunks.append(doc)
                else:
                    # Split text documents
                    split_docs = self.text_splitter.split_documents([doc])
                    chunks.extend(split_docs)
            
            logger.info(f"Created {len(chunks)} total chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        try:
            batch_size = self.config.get("embeddings.batch_size", 32)
            embeddings = []
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    convert_to_tensor=False,
                    show_progress_bar=True if i == 0 else False
                )
                embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def create_document_chunks(
        self, 
        documents: List[Document], 
        doc_metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Create document chunks with sanitized metadata."""
        chunks = []
        
        for idx, doc in enumerate(documents):
            chunk_id = str(uuid.uuid4())
            
            # Extract page number if available
            page_number = doc.metadata.get('page', None)
            if page_number is not None:
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    page_number = None
            
            # 1) Flatten any list-valued metadata entries
            clean_meta = {}
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    clean_meta[key] = ", ".join(map(str, value))
                else:
                    clean_meta[key] = value
            
            # 2) Merge in the standardized fields
            clean_meta.update({
                'source_file': doc_metadata.file_name,
                'file_type': doc_metadata.file_type.value,
                'chunk_index': idx,
                'document_hash': self.get_file_hash(doc_metadata.file_path),
                'processing_timestamp': datetime.now().isoformat()
            })
            
            # 3) Build the chunk
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc_metadata.file_name,
                content=doc.page_content,
                metadata=clean_meta,
                page_number=page_number,
                chunk_index=idx
            )
            chunks.append(chunk)
        
        return chunks

    
    # def create_document_chunks(
    #     self, 
    #     documents: List[Document], 
    #     doc_metadata: DocumentMetadata
    # ) -> List[DocumentChunk]:
    #     """Create document chunks with metadata."""
    #     chunks = []
        
    #     for idx, doc in enumerate(documents):
    #         chunk_id = str(uuid.uuid4())
            
    #         # Extract page number if available
    #         page_number = doc.metadata.get('page', None)
    #         if page_number is not None:
    #             try:
    #                 page_number = int(page_number)
    #             except (ValueError, TypeError):
    #                 page_number = None
            
    #         chunk = DocumentChunk(
    #             chunk_id=chunk_id,
    #             document_id=doc_metadata.file_name,
    #             content=doc.page_content,
    #             metadata={
    #                 **doc.metadata,
    #                 'source_file': doc_metadata.file_name,
    #                 'file_type': doc_metadata.file_type.value,
    #                 'chunk_index': idx,
    #                 'document_hash': self.get_file_hash(doc_metadata.file_path),
    #                 'processing_timestamp': datetime.now().isoformat()
    #             },
    #             page_number=page_number,
    #             chunk_index=idx
    #         )
    #         chunks.append(chunk)
        
    #     return chunks
    
    def process_single_file(self, file_path: str) -> tuple[DocumentMetadata, List[DocumentChunk]]:
        """Process any single file universally."""
        file_name = Path(file_path).name
        logger.info(f"Processing: {file_name}")
        
        # Create metadata
        doc_metadata = self._create_file_metadata(file_path)
        doc_metadata.status = DocumentStatus.PROCESSING
        
        try:
            # Load document (handles any type)
            documents = self.load_document(file_path)
            
            # Split into appropriate chunks
            chunks = self.split_documents(documents)
            
            # Create chunk objects
            document_chunks = self.create_document_chunks(chunks, doc_metadata)
            
            # Generate embeddings
            texts = [chunk.content for chunk in document_chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(document_chunks, embeddings):
                chunk.embedding = embedding
            
            # Update metadata
            doc_metadata.chunk_count = len(document_chunks)
            doc_metadata.processed_at = datetime.now()
            doc_metadata.status = DocumentStatus.COMPLETED
            
            logger.info(f"✅ {file_name} → {len(document_chunks)} chunks")
            return doc_metadata, document_chunks
            
        except Exception as e:
            doc_metadata.status = DocumentStatus.FAILED
            doc_metadata.error_message = str(e)
            logger.error(f"❌ Failed to process {file_name}: {str(e)}")
            raise
    
    def _create_file_metadata(self, file_path: str) -> DocumentMetadata:
        """Create metadata for any file."""
        file_info = os.stat(file_path)
        return DocumentMetadata(
            file_name=Path(file_path).name,
            file_path=file_path,
            file_size=file_info.st_size,
            file_type=self.get_document_type(file_path),
            created_at=datetime.fromtimestamp(file_info.st_ctime),
            status=DocumentStatus.PENDING
        )
    
    def process_all_documents(self) -> tuple[List[DocumentMetadata], List[DocumentChunk]]:
        """Process all documents universally."""
        logger.info(f"🔍 Scanning: {self.data_source_dir}")
        
        all_metadata = []
        all_chunks = []
        
        # Find all supported files
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.csv'}
        data_source_path = Path(self.data_source_dir)
        
        if not data_source_path.exists():
            raise ValueError(f"Data source directory not found: {self.data_source_dir}")
        
        # Collect files from all subdirectories
        files = []
        file_counts = {}
        
        for ext_dir in ['docx','txt','pdf','csv']:
            dir_path = data_source_path / ext_dir
            count = 0
            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        files.append(file_path)
                        count += 1
            file_counts[ext_dir] = count
        
        logger.info(f"📁 Found files: {dict(file_counts)} (Total: {len(files)})")
        
        if not files:
            logger.warning("⚠️ No files found!")
            return all_metadata, all_chunks
        
        # Process each file
        success_count = 0
        for file_path in files:
            try:
                metadata, chunks = self.process_single_file(str(file_path))
                all_metadata.append(metadata)
                all_chunks.extend(chunks)
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ {file_path.name}: {str(e)}")
                continue
        
        # Summary
        logger.info(f"✅ Processed {success_count}/{len(files)} files → {len(all_chunks)} chunks")
        
        # Breakdown by type
        type_summary = {}
        for metadata in all_metadata:
            file_type = metadata.file_type.value
            if file_type not in type_summary:
                type_summary[file_type] = {'files': 0, 'chunks': 0}
            type_summary[file_type]['files'] += 1
            type_summary[file_type]['chunks'] += metadata.chunk_count
        
        for file_type, stats in type_summary.items():
            logger.info(f"   {file_type.upper()}: {stats['files']} files → {stats['chunks']} chunks")
        
        return all_metadata, all_chunks


# For backwards compatibility, create alias
MedicalDocumentProcessor = UniversalDocumentProcessor