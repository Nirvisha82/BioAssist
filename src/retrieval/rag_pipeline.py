"""
RAG Pipeline implementation
"""
from typing import List
from loguru import logger
from src.models.data_models import RetrievalResult, WebSearchResult
from ..ingestion.vector_db_manager import VectorDBInterface
from ..retrieval.web_search import WebSearcher
from ..utils.config_manager import ConfigManager


class RAGPipeline:
    
    def __init__(self, vector_db: VectorDBInterface, config: ConfigManager):
        self.vector_db = vector_db
        self.web_searcher = WebSearcher(config)
        self.config = config
        self.top_k = config.get("retrieval.top_k", 5)
        logger.info("Initialized Minimal RAG Pipeline")
    
    # Only these two methods are actually used by main.py:
    
    def _retrieve_from_kb(self, question: str) -> List[RetrievalResult]:
        #Retrieve relevant documents from knowledge base.
        try:
            results = self.vector_db.similarity_search(question, self.top_k)
            logger.info(f"Retrieved {len(results)} documents from knowledge base")
            return results
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            return []
    
    def _retrieve_from_web(self, question: str) -> List[WebSearchResult]:
        #Retrieve relevant information from web search.
        try:
            docs = self.web_searcher.search(question)
            web_results = []
            for doc in docs:
                web_results.append(
                    WebSearchResult(
                        snippet=doc.page_content,
                        title=doc.metadata.get("title", ""),
                        source=doc.metadata.get("source", "")
                    )
                )
            logger.info(f"Retrieved {len(web_results)} web results")
            return web_results
        except Exception as e:
            logger.error(f"Error retrieving from web: {str(e)}")
            return []