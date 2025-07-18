import streamlit as st
import google.generativeai as genai
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_manager import ConfigManager
from src.utils.helpers import guardrail,rewrite_query
from src.ingestion.vector_db_manager import VectorDBFactory
config = ConfigManager(config_path="config/config.yaml")  # Uses your uploaded YAML

# --- Load config and pipeline ---
@st.cache_resource
def setup_pipeline():
    vector_db = VectorDBFactory.create_vector_db(config)
    pipeline = RAGPipeline(vector_db, config)
    return pipeline

# --- Setup Gemini ---
@st.cache_resource
def setup_gemini():
    gemini_api_key = config.get("llm.api_key")
    model=config.get("llm.model_name")
    print("Model : ",model," ",gemini_api_key)
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel(model)

# --- Format prompt with retrieved chunks + chat history ---
def build_prompt(chat_history, query, kb_chunks, web_chunks,re_written_query=""):
    KB_context = "\n".join([f"• {chunk.content}" for chunk in kb_chunks])
    WEB_context = "\n".join([f"• {chunk.snippet}" for chunk in web_chunks])
    history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in chat_history])
    return f"""
[System]:
You are BioAssist, a friendly biomedical AI assistant.
Always answer using both the local document store and live web search.
Provide each facts' reference you use with a number in bracket after fact. And at the cite them
-Local docs as [1]Doc: <document_name>
-Web sources as [2]Web: <url>
If both sources are relevant, give them equal weight in your answer.
If only one is relevant, rely on it—but still explain which source it came from. W
Keep your tone conversational, but stick strictly to biomedical Q&A.

[Chat History]:
{history}

[Local Context]:
{KB_context}

[Web Context]:
{WEB_context}

[Instructions]:
Use both contexts to answer.
Cite every fact like (Doc: TitleXYZ) or (Web: https://…).
Be concise and conversational.

[Example]:
Example 1:
Q: What is the mechanism of action of tocilizumab?
A: Tocilizumab is a monoclonal antibody that blocks the interleukin‐6 (IL‑6) receptor on immune cells[1], preventing IL‑6–mediated pro‑inflammatory signaling[2] 
[1]Doc: “Cytokine Therapies”
[2]Web: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456)

Example 2:
Q: Can I use metformin in patients with severe renal impairment?
A: Metformin is excreted unchanged by the kidneys[1], so in severe renal impairment (eGFR < 30 mL/min/1.73 m²)[2] it’s contraindicated due to lactic acidosis risk 
[1]Doc: “Renal Drug Handbook”
[2]Web: https://www.fda.gov/drugs/drug-safety-and-availability)


Now, answer the user’s question below…
User Question: {query}
Answer:
"""

# --- Streamlit UI ---
st.set_page_config(page_title="Bio Assist")
st.title("🧠Bio Assist - A Biomedical Web Search Enabled RAG Chatbot")

# Setup once
pipeline = setup_pipeline()
gemini_model = setup_gemini()

# Session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question...")

if query:
    with st.spinner("Retrieving context and generating answer..."):
        rewritten_query = rewrite_query(gemini_model, st.session_state.chat_history, query)
        print(f"New Query: {rewritten_query}")
        if not guardrail(gemini_model, rewritten_query):
            st.session_state.chat_history.append({
                "user": query,
                "bot": "Oops, my microscopes are all busy—can’t help with that one"
            })
        else:
            # # Step 1: Retrieve from KB and Web
            kb_chunks = pipeline._retrieve_from_kb(rewritten_query)
            web_chunks = pipeline._retrieve_from_web(rewritten_query)
            # kb_chunks = pipeline._retrieve_from_kb(query)
            # web_chunks = pipeline._retrieve_from_web(query)

            # Step 2: Construct prompt
            prompt = build_prompt(st.session_state.chat_history, query, kb_chunks, web_chunks,rewritten_query)

            # Step 3: Generate answer
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()

            # Step 4: Update chat memory
            st.session_state.chat_history.append({
                "user": query,
                "bot": answer
            })

# --- Display Chat ---
for turn in st.session_state.chat_history[::-1]:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["bot"])

# # --- Optional Debug Info ---
# with st.expander("📄 Retrieved Sources"):
#     st.subheader("📚 Local KB Chunks")
#     for chunk in kb_chunks:
#         st.markdown(f"**Source:** `{chunk.source_document}`\n\n{chunk.content[:300]}...")

#     st.subheader("🌐 Web Search Results")
#     for chunk in web_chunks:
#         st.markdown(f"**URL:** {chunk.url}\n\n{chunk.snippet[:300]}...")

