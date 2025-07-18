import streamlit as st
import google.generativeai as genai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_manager import ConfigManager
from src.ingestion.vector_db_manager import VectorDBFactory

config = ConfigManager(config_path="config/config.yaml")

# --- Load config and pipeline ---
@st.cache_resource
def setup_pipeline():
    vector_db = VectorDBFactory.create_vector_db(config)
    pipeline = RAGPipeline(vector_db, config)
    return pipeline

# --- Setup Gemini with LangChain ---
@st.cache_resource
def setup_gemini_langchain():
    gemini_api_key = config.get("llm.api_key")
    model_name = config.get("llm.model_name")
    max_tokens = config.get("llm.max_tokens", 1000)  # Default to 1000 if not specified
    temperature = config.get("llm.temperature", 0.7)  # Default to 0.7 if not specified
    
    print(f"Model: {model_name}, Max Tokens: {max_tokens}, Temperature: {temperature}")
    
    # Configure genai
    genai.configure(api_key=gemini_api_key)
    
    # Create LangChain-compatible Gemini LLM
    llm = GoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    
    return llm

# --- Setup Conversational Retrieval Chain ---
@st.cache_resource
def setup_conversational_chain():
    pipeline = setup_pipeline()
    llm = setup_gemini_langchain()
    
    # Create memory for conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create a combined retriever that uses both KB and Web
    class CombinedRetriever:
        def __init__(self, pipeline):
            self.pipeline = pipeline
        
        def get_relevant_documents(self, query):
            # Retrieve from both KB and Web
            kb_chunks = self.pipeline._retrieve_from_kb(query)
            web_chunks = self.pipeline._retrieve_from_web(query)
            
            # Convert to LangChain Document format
            from langchain.schema import Document
            
            documents = []
            # Add KB chunks
            for chunk in kb_chunks:
                documents.append(Document(
                    page_content=chunk.content,
                    metadata={"source": "knowledge_base", "type": "kb"}
                ))
            
            # Add web chunks
            for chunk in web_chunks:
                documents.append(Document(
                    page_content=chunk.snippet,
                    metadata={"source": getattr(chunk, 'source', 'web'), "type": "web"}
                ))
            
            return documents
    
    retriever = CombinedRetriever(pipeline)
    
    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return chain

# --- Alternative: Manual setup with proper Gemini configuration ---
@st.cache_resource
def setup_gemini_manual():
    gemini_api_key = config.get("llm.api_key")
    model_name = config.get("llm.model_name")
    max_tokens = config.get("llm.max_tokens", 1000)
    temperature = config.get("llm.temperature", 0.7)
    
    print(f"Model: {model_name}, Max Tokens: {max_tokens}, Temperature: {temperature}")
    
    genai.configure(api_key=gemini_api_key)
    
    # Create generation config
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )
    
    return model

# --- Format prompt with retrieved chunks + chat history (for manual approach) ---
def build_prompt_with_history(chat_history, query, kb_chunks, web_chunks):
    KB_context = "\n".join([f"• {chunk.content}" for chunk in kb_chunks])
    WEB_context = "\n".join([f"• {chunk.snippet}" for chunk in web_chunks])
    
    # Format chat history properly
    history_text = ""
    for turn in chat_history:
        history_text += f"Human: {turn['user']}\nAssistant: {turn['bot']}\n\n"
    
    return f"""
You are a helpful AI assistant. Use the following context to answer the user query. Be concise and include references to the sources.

Previous Conversation:
{history_text}

Knowledge Base Context:
{KB_context}

Web Context:
{WEB_context}

Current Question: {query}

Please provide a helpful answer based on the context and conversation history:
"""

# --- Streamlit UI ---
st.set_page_config(page_title="RAG + Web + Gemini Chatbot")
st.title("🧠 Web-Enabled RAG Chatbot with Gemini")

# Choose approach
approach = st.sidebar.selectbox(
    "Select Approach:",
    ["LangChain Conversational Chain", "Manual with Proper Config"]
)

# Setup based on approach
if approach == "LangChain Conversational Chain":
    try:
        chain = setup_conversational_chain()
        use_langchain = True
    except Exception as e:
        st.error(f"Error setting up LangChain approach: {e}")
        st.info("Falling back to manual approach...")
        use_langchain = False
        pipeline = setup_pipeline()
        gemini_model = setup_gemini_manual()
else:
    use_langchain = False
    pipeline = setup_pipeline()
    gemini_model = setup_gemini_manual()

# Session memory (for manual approach)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question...")

if query:
    with st.spinner("Retrieving context and generating answer..."):
        if use_langchain and 'chain' in locals():
            # LangChain approach
            try:
                result = chain({"question": query})
                answer = result["answer"]
                
                # Update session state for display
                st.session_state.chat_history.append({
                    "user": query,
                    "bot": answer
                })
            except Exception as e:
                st.error(f"Error with LangChain approach: {e}")
                # Fallback to manual
                use_langchain = False
        
        if not use_langchain:
            # Manual approach with proper config
            kb_chunks = pipeline._retrieve_from_kb(query)
            web_chunks = pipeline._retrieve_from_web(query)
            
            prompt = build_prompt_with_history(
                st.session_state.chat_history, query, kb_chunks, web_chunks
            )
            
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()
            
            st.session_state.chat_history.append({
                "user": query,
                "bot": answer
            })

# --- Display Chat (Fixed: Most recent at bottom) ---
for turn in st.session_state.chat_history:  # Removed [::-1] to show chronological order
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["bot"])

# --- Clear Chat Button ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    if use_langchain and 'chain' in locals():
        # Clear LangChain memory
        chain.memory.clear()
    st.rerun()