"""
Data models for the RAG chatbot system.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    XLSX = "xlsx"
    PPTX = "pptx"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    file_name: str
    file_path: str
    file_size: int
    file_type: DocumentType
    created_at: datetime
    processed_at: Optional[datetime] = None
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: Optional[str] = None


class DocumentChunk(BaseModel):
    """Document chunk model."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_index: int
    embedding: Optional[List[float]] = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    session_id: str
    user_id: Optional[str] = None
    include_web_search: bool = True
    max_results: int = 5
    timestamp: datetime = Field(default_factory=datetime.now)


class RetrievalResult(BaseModel):
    """Retrieval result model."""
    chunk_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_document: str


class WebSearchResult(BaseModel):
    """Web search result model."""
    snippet: str
    title: str
    source: str
    # relevance_score: float
    # url: str


class ChatMessage(BaseModel):
    """Chat message model."""
    message_id: str
    session_id: str
    user_message: str
    bot_response: str
    retrieved_chunks: List[RetrievalResult]
    web_results: List[WebSearchResult]
    timestamp: datetime
    processing_time: float
    confidence_score: float


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime
    total_documents: int
    total_chunks: int
    avg_query_time: float
    total_queries: int
    error_rate: float
    vector_db_size: int
    memory_usage: float
    cpu_usage: float


class SafetyCheck(BaseModel):
    """Safety check result."""
    is_safe: bool
    confidence: float
    reason: Optional[str] = None
    suggested_action: Optional[str] = None