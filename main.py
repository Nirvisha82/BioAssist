import os
import json
import time
import streamlit as st
import google.generativeai as genai
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_manager import ConfigManager
from src.utils.helpers import guardrail, rewrite_query, verify_grounding
from src.ingestion.vector_db_manager import VectorDBFactory
from loguru import logger

# Setup pipeline & model
config = ConfigManager(config_path="config/config.yaml")

STORAGE_DIR = "data"
CONVO_FILE = os.path.join(STORAGE_DIR, "conversations.json")
MIN_LOCAL_SIMILARITY = config.get("retrieval.similarity_threshold")  # threshold for local document relevance

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

# Persistence helpers
def load_conversations():
    try:
        with open(CONVO_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_conversations(store):
    with open(CONVO_FILE, "w") as f:
        json.dump(store, f, indent=4)

# --- Session state init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_store" not in st.session_state:
    st.session_state.conversation_store = load_conversations()
    logger.info("Loaded conversation from local storage.")
# Assign a fresh active_convo_id when starting or after clearing
if "active_convo_id" not in st.session_state or not st.session_state.chat_history:
    st.session_state.active_convo_id = f"convo{len(st.session_state.conversation_store)+1}"


@st.cache_resource
def setup_pipeline():
    return RAGPipeline(VectorDBFactory.create_vector_db(config), config)

@st.cache_resource
def setup_gemini():
    genai.configure(api_key=config.get("llm.api_key"))
    return genai.GenerativeModel(config.get("llm.model_name"))

pipeline = setup_pipeline()
gemini_model = setup_gemini()
logger.info("Pipeline Setup Complete")

# Prompt builder
def build_prompt(chat_history, query, kb_chunks, web_chunks, re_written_query=""):
    KB_ctx = "\n".join([f"• {c.content}" for c in kb_chunks])
    WEB_ctx = "\n".join([f"• {w.snippet}" for w in web_chunks])
    hist   = "\n".join([f"User: {t['user']}\nAssistant: {t['bot']}" for t in chat_history])
    return f"""
[System]:
You are BioAssist, a biomedical AI assistant.
Use both local documents and live web search to provide a detailed answer unless specified otherwise.


[Chat History]:
{hist}

[Local Context]:
{KB_ctx}

[Web Context]:
{WEB_ctx}

User Question: {query}
Answer:
"""

# <--------------------- Streamlit layout ------------->
st.set_page_config(page_title="Bio Assist")
st.markdown('<div class="big-title">🧠 BioAssist - Your Biomedical Web‑Enabled RAG Chatbot</div>', unsafe_allow_html=True)

# --- Custom CSS to tighten the UI ---
st.markdown("""
<style>
/* Smaller, bolder app title */
.big-title {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 4px;
}
/* Subtitle under the title */
.small-subheader {
    font-size: 20px;
    margin-top: -8px;
    margin-bottom: 16px;
}
/* Tighter buttons everywhere */
.stButton > button {
    padding: 4px 8px !important;
    font-size: 14px !important;
}
/* Narrow the download button in the sidebar */
.stDownloadButton > button {
    width: 100% !important;
    padding: 4px 8px !important;
    font-size: 14px !important;
            

}
</style>
""", unsafe_allow_html=True)

# Main chat input
query = st.chat_input("Ask a question...")
if query:
    with st.spinner("Processing..."):
        t0 = time.time()

        # rewrite
        rewritten = rewrite_query(gemini_model, st.session_state.chat_history, query)

        # init placeholders
        sources = []
        metrics = {}
        fallback_note = None


        SELF_QUES = "The user is asking about BioAssist Chatbot about myself."
        if rewritten.strip() == SELF_QUES:
            # Let the LLM generate a natural greeting
            greeting_prompt = f"""
            You are BioAssist, a friendly biomedical AI assistant.
            The user just asked: "{query}"
            Please reply in a warm, conversational tone—no citations or web searches.
            """
            t1 = time.time()
            res = gemini_model.generate_content(greeting_prompt)
            t2 = time.time()
            response_text = res.text.strip()

            # record minimal metrics and no sources
            sources = []
            metrics = {
                "generation_time": round(t2 - t1, 3),
                "prompt_len": len(greeting_prompt),
                "response_tokens": len(response_text.split()),
            }

        # Non‑medical scope guardrail
        elif not guardrail(gemini_model, rewritten):
            response_text = "I'm sorry, I can only answer biomedical queries."
            kb_chunks = []
            web_chunks = []

        else:
            # Relevant not interaction query.
            # local retrieval
            kb_chunks = pipeline._retrieve_from_kb(rewritten)

            # evaluate best local score
            best_score = max((c.similarity_score for c in kb_chunks), default=0.0)
            if best_score < MIN_LOCAL_SIMILARITY:
                st.info(f"🔎 Local matches below {MIN_LOCAL_SIMILARITY:.2f}; falling back to web search.")
                kb_chunks = []
                web_chunks = pipeline._retrieve_from_web(rewritten)
                fallback_note = "Used web search fallback."
            else:
                # filter out low-confidence docs
                kb_chunks = [c for c in kb_chunks if c.similarity_score >= MIN_LOCAL_SIMILARITY]
                web_chunks = pipeline._retrieve_from_web(rewritten)
                fallback_note = None

            # record sources
            for c in kb_chunks:
                sources.append(("Local", c.source_document, c.similarity_score))
            for w in web_chunks:
                sources.append(("Web", w.source, None))

            # build prompt & call LLM
            prompt = build_prompt(
                st.session_state.chat_history,
                query,
                kb_chunks,
                web_chunks,
                rewritten
            )
            t1 = time.time()
            res = gemini_model.generate_content(prompt)
            t2 = time.time()
            response_text = res.text.strip()
            contexts = [c.content for c in kb_chunks] + [w.snippet for w in web_chunks]
            # unsupported= verify_grounding(response_text, contexts,gemini_model)
           

            # if unsupported != []:
            #     for uns in unsupported:
            #         print(uns)
            #     st.warning("⚠️ I’m not fully certain about these statements. Fact Check them.")
        
            # record metrics
            metrics = {
                "retrieval_time": round(t1 - t0, 3),
                "generation_time": round(t2 - t1, 3),
                "prompt_len": len(prompt),
                "response_tokens": len(response_text.split()),
            }

        # append and persist
        turn = {
            "user": query,
            "bot": response_text,
            "sources": sources,
            "metrics": metrics,
            "note": fallback_note,
        }
        st.session_state.chat_history.append(turn)
        st.session_state.conversation_store[
            st.session_state.active_convo_id
        ] = st.session_state.chat_history.copy()
        save_conversations(st.session_state.conversation_store)


# Sidebar: past conversations
st.sidebar.title("📂 Past Conversations")
# store = load_conversations()
store = st.session_state.conversation_store
if store:
    for cid, convo in store.items():
        title = convo[0]['user'] if convo else cid
        with st.sidebar.expander(title, expanded=False):
            for t in convo:
                st.markdown(f"**User:** {t['user']}")
                st.markdown(f"**Bot:** {t['bot']}")
            cols = st.sidebar.columns([1,1])
            with cols[0]:
                st.download_button(
                    label="Export Chat",
                    data="\n\n".join([f"User: {x['user']}\nBot: {x['bot']}" for x in convo]),
                    file_name=f"{title}.txt",
                    mime="text/plain",
                    key=f"exp_{cid}",
                    use_container_width=True
                )
            with cols[1]:
                if st.button(
                    "Delete Chat",
                    key=f"del_{cid}",
                    use_container_width=True
                ):
                    st.session_state.conversation_store.pop(cid, None)
                    #persist to disk
                    save_conversations(st.session_state.conversation_store)
                    #stop this loop so the sidebar re‑renders without this expander
                    break
else:
    st.sidebar.info("No previous conversations yet.")

# Chat controls
st.markdown('<div class="small-subheader">⚙️ Chat Controls</div>', unsafe_allow_html=True)
ctrl_cols = st.columns([2,1,1])

with ctrl_cols[0]:
    # only show small selector + button
    txt = "\n\n".join([f"User: {x['user']}\nBot: {x['bot']}" for x in st.session_state.chat_history])
    fmt = st.selectbox(
        "BioAssist",    
        [".txt", ".md"],
        key="exp_fmt",
        label_visibility="collapsed"
    )


with ctrl_cols[2]:
    if st.button("End Chat", key="end_chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.active_convo_id = f"convo{len(load_conversations())+1}"

with ctrl_cols[1]:
    st.download_button(
    label="Export Chat",
    data=txt.encode("utf-8"),
    file_name=f"chat_history{fmt}",
    mime="text/plain",
    use_container_width=True
)

# Display current conversation
st.markdown('<div class="small-subheader">💬 Current Conversation</div>', unsafe_allow_html=True)
for t in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(t['user'])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(t['bot'])
        # display sources
        if t.get('sources'):
            st.markdown("**Sources:**")
            for s in t['sources']:
                if s[0] == "Local":
                    st.markdown(f"- Local: {s[1]} (score {s[2]:.2f})")
                else:
                    st.markdown(f"- Web: {s[1]}")
        # display metrics
        with st.expander("Performance Metrics", expanded=False):
            for k, v in t['metrics'].items():
                st.write(f"{k}: {v}")
        # fallback note
        if t.get('note'):
            st.info(t['note'])
