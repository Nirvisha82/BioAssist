"""
RAG Pipeline implementation with LLM integration.
"""
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

from langchain.llms import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from loguru import logger
from src.models.data_models import WebSearchResult

from ..models.data_models import (
    QueryRequest, ChatMessage, RetrievalResult, 
    WebSearchResult, SafetyCheck
)
from ..ingestion.vector_db_manager import VectorDBInterface
from ..retrieval.web_search import WebSearcher
from ..utils.config_manager import ConfigManager


class LLMManager:
    """Manages LLM interactions and configurations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.provider = config.get("llm.provider", "google")
        self.model_name = config.get("llm.model_name", "gemini-1.5-flash")
        self.temperature = config.get("llm.temperature", 0.1)
        self.max_tokens = config.get("llm.max_tokens", 2000)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        logger.info(f"Initialized LLM: {self.provider}/{self.model_name}")
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        try:
            if self.provider == "google":
                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.config.get("llm.top_p", 0.95)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, system_message: str = None) -> str:
        """Generate response from LLM."""
        try:
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise


class PromptManager:
    """Manages prompts for different RAG scenarios."""
    
    def __init__(self):
        self.system_prompts = {
            "rag_response": """You are a knowledgeable AI assistant specializing in climate change and environmental policy. 
Your role is to provide accurate, well-researched answers based on the provided context from documents and web sources.

Guidelines:
1. Always base your answers on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sources when making claims
4. Be objective and balanced in your responses
5. If information conflicts between sources, acknowledge this
6. Provide actionable insights when appropriate
7. Use clear, accessible language while maintaining accuracy""",

            "no_context": """You are a helpful AI assistant. The user has asked a question but no relevant context was found in the knowledge base or web search. 
Provide a brief, helpful response acknowledging this limitation and suggest how the user might find the information they need.""",

            "safety_check": """Analyze the following query for potential safety concerns:
- Harmful or misleading information requests
- Attempts to generate false climate information
- Requests for dangerous activities
- Privacy or security concerns

Return 'SAFE' if the query is appropriate, or 'UNSAFE' with a brief explanation if not."""
        }
        
        self.rag_template = """Context from Knowledge Base:
{kb_context}

Context from Web Search:
{web_context}

Question: {question}

Based on the provided context, please provide a comprehensive and accurate answer. 
If you need to make any assumptions or if the context is insufficient, please state this clearly.
Include relevant source references in your response.

Answer:"""

        self.conversational_template = """Previous conversation:
{chat_history}

Current context:
{context}

Question: {question}

Please provide a response that takes into account the conversation history and current context.

Answer:"""
    
    def get_rag_prompt(self, question: str, kb_context: str, web_context: str = "") -> str:
        """Generate RAG prompt with context."""
        return self.rag_template.format(
            question=question,
            kb_context=kb_context,
            web_context=web_context
        )
    
    def get_conversational_prompt(self, question: str, context: str, chat_history: str = "") -> str:
        """Generate conversational prompt with history."""
        return self.conversational_template.format(
            question=question,
            context=context,
            chat_history=chat_history
        )


class SafetyGuards:
    """Safety mechanisms for the RAG system."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.blocked_patterns = [
            r"how to.*harm",
            r"create.*weapon",
            r"illegal.*activity",
            r"false.*climate.*data"
        ]
    
    def check_query_safety(self, query: str) -> SafetyCheck:
        """Check if a query is safe to process."""
        try:
            # Basic pattern matching
            import re
            for pattern in self.blocked_patterns:
                if re.search(pattern, query.lower()):
                    return SafetyCheck(
                        is_safe=False,
                        confidence=0.9,
                        reason="Query contains potentially harmful patterns",
                        suggested_action="Rephrase your question to focus on factual information"
                    )
            
            # LLM-based safety check for complex cases
            safety_prompt = f"""Analyze this query for safety concerns: "{query}"
            
            Is this query safe and appropriate? Respond with only 'SAFE' or 'UNSAFE: reason'"""
            
            response = self.llm_manager.generate_response(safety_prompt)
            
            if response.startswith("UNSAFE"):
                return SafetyCheck(
                    is_safe=False,
                    confidence=0.8,
                    reason=response.split(":", 1)[1].strip() if ":" in response else "Query flagged as unsafe",
                    suggested_action="Please rephrase your question"
                )
            
            return SafetyCheck(is_safe=True, confidence=0.9)
            
        except Exception as e:
            logger.error(f"Error in safety check: {str(e)}")
            return SafetyCheck(is_safe=True, confidence=0.5, reason="Safety check failed")
    
    def check_response_quality(self, response: str, context: str) -> float:
        """Check response quality and potential hallucination."""
        try:
            # Simple quality checks
            quality_score = 1.0
            
            # Check if response is too short
            if len(response) < 50:
                quality_score -= 0.2
            
            # Check if response acknowledges context
            if context and "based on" not in response.lower() and "according to" not in response.lower():
                quality_score -= 0.1
            
            # Check for common hallucination patterns
            hallucination_patterns = [
                "I don't have access to",
                "I cannot browse",
                "I don't have real-time",
                "As an AI language model"
            ]
            
            for pattern in hallucination_patterns:
                if pattern.lower() in response.lower():
                    quality_score -= 0.1
            
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error checking response quality: {str(e)}")
            return 0.5


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(
        self,
        vector_db: VectorDBInterface,
        config: ConfigManager
    ):
        self.vector_db     = vector_db
        self.web_searcher  = WebSearcher(config)
        self.config        = config
        
        # Initialize components
        self.llm_manager = LLMManager(config)
        self.prompt_manager = PromptManager()
        self.safety_guards = SafetyGuards(self.llm_manager)
        
        # Configuration
        self.top_k = config.get("retrieval.top_k", 5)
        self.enable_web_search = config.get("web_search.enabled", True)
        self.enable_safety = config.get("safety.enable_guardrails", True)
        
        logger.info("Initialized RAG Pipeline")
    
    def process_query(self, query_request: QueryRequest) -> ChatMessage:
        """Process a query through the complete RAG pipeline."""
        start_time = time.time()
        
        try:
            # Safety check
            if self.enable_safety:
                safety_check = self.safety_guards.check_query_safety(query_request.question)
                if not safety_check.is_safe:
                    return self._create_safety_response(query_request, safety_check)
            
            # Retrieve from knowledge base
            kb_results = self._retrieve_from_kb(query_request.question)
            
            # Determine if web search is needed
            web_results = []
            confidence_score = self._calculate_confidence(kb_results)
            
            if (self.enable_web_search and 
                query_request.include_web_search and 
                self.web_searcher.should_search_web(kb_results, confidence_score)):
                web_results = self._retrieve_from_web(query_request.question)
            
            # Generate response
            response = self._generate_response(
                query_request.question, 
                kb_results, 
                web_results
            )
            
            # Quality check
            if self.enable_safety:
                quality_score = self.safety_guards.check_response_quality(
                    response, 
                    self._format_context(kb_results, web_results)
                )
                confidence_score = min(confidence_score, quality_score)
            
            # Create chat message
            processing_time = time.time() - start_time
            
            chat_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=query_request.session_id,
                user_message=query_request.question,
                bot_response=response,
                retrieved_chunks=kb_results,
                web_results=web_results,
                timestamp=datetime.now(),
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with confidence {confidence_score:.2f}")
            return chat_message
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._create_error_response(query_request, str(e))
    
    def _retrieve_from_kb(self, question: str) -> List[RetrievalResult]:
        """Retrieve relevant documents from knowledge base."""
        try:
            results = self.vector_db.similarity_search(question, self.top_k)
            logger.info(f"Retrieved {len(results)} documents from knowledge base")
            return results
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            return []
    
    def _retrieve_from_web(self, question: str) -> List[WebSearchResult]:
            """Retrieve relevant information from web search."""
            docs = self.web_searcher.search(question)
            print(f"Got {len(docs)} web results for :  {question}")
            web_results = []
            for doc in docs:
                # doc.metadata now contains both source (URL) and title
                print(doc)
                web_results.append(
                    WebSearchResult(
                        snippet=doc.page_content,
                        title=doc.metadata.get("title", ""),
                        source=doc.metadata.get("source", "")
                        # relevance_score=0.2  ,              # or use a real score if you have it
                    )
                )
            logger.info(f"Retrieved {len(web_results)} web results")
            return web_results
    
    def _calculate_confidence(self, kb_results: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not kb_results:
            return 0.0
        
        # Average similarity score
        avg_similarity = sum(result.similarity_score for result in kb_results) / len(kb_results)
        
        # Boost for multiple high-quality results
        high_quality_count = sum(1 for result in kb_results if result.similarity_score > 0.8)
        quality_boost = min(0.2, high_quality_count * 0.05)
        
        return min(1.0, avg_similarity + quality_boost)
    
    def _format_context(self, kb_results: List[RetrievalResult], web_results: List[WebSearchResult]) -> str:
        """Format context from retrieval results."""
        context_parts = []
        
        # Knowledge base context
        if kb_results:
            kb_context = "Knowledge Base Sources:\n"
            for i, result in enumerate(kb_results, 1):
                kb_context += f"{i}. {result.source_document} (Score: {result.similarity_score:.2f})\n"
                kb_context += f"   {result.content[:300]}...\n\n"