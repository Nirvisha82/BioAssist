import os
import json
import time
import streamlit as st
import google.generativeai as genai
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_manager import ConfigManager
from src.utils.helpers import guardrail, rewrite_query, verify_grounding, split_sentences
from src.ingestion.vector_db_manager import VectorDBFactory
from loguru import logger

# Setup pipeline & model
config = ConfigManager(config_path="config/config.yaml")

STORAGE_DIR = "conversation_data"
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
def build_prompt(chat_history, query, kb_chunks, web_chunks, re_written_query):
    KB_ctx = "\n".join([f"• {c.content}" for c in kb_chunks])
    WEB_ctx = "\n".join([f"• {w.snippet}" for w in web_chunks])
    hist   = "\n".join([f"User: {t['user']}\nAssistant: {t['bot']}" for t in chat_history])
    return f"""
[System]:
You are BioAssist, a biomedical AI assistant.
Use both local documents and live web search to provide a detailed answer unless specified otherwise.
If you are giving a question that is no-where related to you or medical field, deny the answer politely and re-state your capabilities.


[Chat History]:
{hist}

[Local Context]:
{KB_ctx}

[Web Context]:
{WEB_ctx}

User Question: {re_written_query}
Answer:
"""

# <--------------------- Streamlit layout ------------->
st.set_page_config(page_title="Bio Assist")
st.markdown('<div class="big-title">🧠 BioAssist - Biomedical Web‑Search Enabled RAG Chatbot</div>', unsafe_allow_html=True)

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

def is_greeting_or_self_reference(query_text):
            # Clean the query by removing punctuation and converting to lowercase
            import re
            query_clean = re.sub(r'[!?.,]', '', query_text.lower().strip())
            
            # Common greetings (without punctuation)
            greetings = [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "thank you", "thanks", "bye", "goodbye", "how are you", "how are you doing"
            ]
            
            # Self-reference questions (without punctuation)
            self_reference = [
                "who are you", "what are you", "what can you do", "what do you do",
                "tell me about yourself", "introduce yourself", "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "thank you", "thanks", "bye", "goodbye", "how are you", "how are you doing"
            ]
            
            # Check if cleaned query matches any greeting or self-reference
            if query_clean in greetings + self_reference:
                return True
            
            # Check if query contains "bioassist" (referring to the bot)
            if "bioassist" in query_text.lower():
                return True
                
            return False

query = st.chat_input("Ask a question...")
if query:
    with st.spinner("Processing..."):
        t0 = time.time()

        # initialize
        sources = []
        metrics = {}
        fallback_note = None
        response_text = None

        if is_greeting_or_self_reference(query):
            greeting_prompt = f"""
            You are BioAssist, a friendly biomedical AI assistant.
            The user just said: \"{query}\"
            Please reply warmly and introduce yourself as a biomedical AI assistant that helps with healthcare and medical questions.
            Keep it conversational and friendly.
            """
            t1 = time.time()
            res = gemini_model.generate_content(greeting_prompt)
            t2 = time.time()
            response_text = res.text.strip()

            kb_chunks = []
            web_chunks = []
            sources = []
            fallback_note = None
            metrics = {
                "generation_time": round(t2 - t1, 3),
                "prompt_len": len(greeting_prompt),
                "response_tokens": len(response_text.split()),
                "ungrounded_statements": 0,
                "grounding_ratio": 1.0,
                "hallucination_flag": False
            }
        # STEP 2: Apply guardrail to ORIGINAL query with chat history context
        elif not guardrail(gemini_model, query, st.session_state.chat_history):
            response_text = "⚠️ Sorry, I can only answer biomedical, clinical, or healthcare-related questions."
            kb_chunks = []
            web_chunks = []
            sources = []
            fallback_note = None
            metrics = {
                "generation_time": 0.0,
                "prompt_len": 0,
                "response_tokens": 0,
                "ungrounded_statements": 0,
                "grounding_ratio": 1.0,
                "hallucination_flag": False
            }

        # STEP 3: Normal biomedical query - rewrite for retrieval
        else:
            # Now rewrite the query for better retrieval (since we know it's medical)
            rewritten = rewrite_query(gemini_model, st.session_state.chat_history, query)
            
            kb_chunks = pipeline._retrieve_from_kb(rewritten)
            best_score = max((c.similarity_score for c in kb_chunks), default=0.0)

            if best_score < MIN_LOCAL_SIMILARITY:
                st.info(f"🔎 Local matches below {MIN_LOCAL_SIMILARITY:.2f}; falling back to web search.")
                kb_chunks = []
                web_chunks = pipeline._retrieve_from_web(rewritten)
                fallback_note = "Used web search fallback."
            else:
                kb_chunks = [c for c in kb_chunks if c.similarity_score >= MIN_LOCAL_SIMILARITY]
                web_chunks = pipeline._retrieve_from_web(rewritten)
                fallback_note = None

            for c in kb_chunks:
                sources.append(("Local", c.source_document, c.similarity_score))
            for w in web_chunks:
                sources.append(("Web", w.source, None))

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
            unsupported = verify_grounding(response_text, contexts, gemini_model)
            total_sentences = len(split_sentences(response_text))
            grounding_ratio = 1 - (len(unsupported) / max(1, total_sentences))
            hallucination_flag = grounding_ratio < 0.85

            if hallucination_flag:
                st.warning("⚠️ BioAssist detected some potentially ungrounded statements:")
                for idx, sent in enumerate(unsupported, 1):
                    st.markdown(f"✖️ {sent}")
            else:
                st.success("✅ All statements appear grounded.")

            metrics = {
                "retrieval_time": round(t1 - t0, 3),
                "generation_time": round(t2 - t1, 3),
                "prompt_len": len(prompt),
                "response_tokens": len(response_text.split()),
                "ungrounded_statements": len(unsupported),
                "grounding_ratio": round(grounding_ratio, 2),
                "hallucination_flag": hallucination_flag
            }

        # Record conversation turn
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

        if t['metrics'].get("hallucination_flag"):
            st.error("Hallucination detected in the response.")

        # fallback note
        if t.get('note'):
            st.info(t['note'])
