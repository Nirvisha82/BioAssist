from transformers import pipeline
import re
from loguru import logger

nli = pipeline("text-classification", model="roberta-large-mnli")

def guardrail(gemini_model, query, chat_history=None):
    # Build context from chat history if available
    context = ""
    if chat_history:
        recent_context = chat_history[-3:]  # Last 3 exchanges for context
        context = "\n".join([f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in recent_context])
    
    classification_prompt = f"""
You are a binary classifier for a biomedical AI assistant.

Chat History (for context):
{context}

Current Question: "{query}"

Rules:
- Answer "Yes" if the question is about healthcare, biomedical science, clinical topics, medicine, or about the assistant itself
- Answer "Yes" for follow-up questions that clearly refer to previous medical topics (using "it", "this", "that", etc.)
- Answer "Yes" for greetings, thank you messages, or questions about the assistant's capabilities
- Answer "No" for questions about non-medical topics (celebrities, sports, politics, general knowledge, etc.)

Examples:
- "What is diabetes?" → Yes (medical topic)
- "How can we treat it?" (after discussing diabetes) → Yes (medical follow-up)
- "Who is Taylor Swift?" → No (celebrity/entertainment)
- "What's the weather like?" → No (general knowledge)

Respond with **only** "Yes" or "No".

Answer (Yes or No only):
"""
    response = gemini_model.generate_content(classification_prompt)
    return response.text.strip().lower() == "yes"


def rewrite_query(gemini_model, chat_history, query):
    history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in chat_history])
    rewrite_prompt = f"""
Given the conversation below and the latest user question, rewrite the question so that it is fully self‑contained \
    and does not use pronouns or ambiguous references.

If the question is asking about the assistant itself (e.g. "Who are you?", \
    "What can you do?", simple greetings or thank‑yous):\
        Respond only with: "The user is asking about BioAssist Chatbot about myself."

Otherwise, rewrite the question to be self-contained while preserving its original intent and topic. \
Do NOT change the subject matter or force it into a medical context if it's not medical.

This rewritten question will be used for guardrail checking and search. Only provide the rewritten question and nothing else in your response.

Chat:
{history}

Latest Question: "{query}"

Rewritten Question:
"""
    response = gemini_model.generate_content(rewrite_prompt)
    logger.info(f"Re-wrote {query} as {response.text.strip()}")
    return response.text.strip()


def split_sentences(text):
    # very basic sentence splitter
    return re.split(r'(?<=[.!?])\s+', text.strip())

import json

def verify_grounding(answer: str, contexts: list[str], gemini_model) -> list[str]:
    """
    Ask the LLM itself to flag any sentences in `answer` 
    that aren’t directly supported by the concatenated `contexts`.
    Returns a list of unsupported sentences (or [] if all are fine).
    """
    import json
    import re
    from loguru import logger

    prompt = f"""
    You are a biomedical fact-checker. Below is the assistant’s answer and the supporting context (retrieved from documents and web search).

    Context:
    {'\n\n'.join(contexts)}

    Answer:
    {answer}

    Your job is to identify only the sentences in the answer that introduce specific factual claims *not supported* by the context above.

    ❗ Do NOT flag general definitions, background information, or common medical knowledge unless they contradict the context.

    Only include unsupported *claims* — not harmless generalizations.

    Return the result as a clean JSON array of sentences. If every sentence is grounded or harmless, return [].
    """


    resp = gemini_model.generate_content(prompt)
    text = resp.text.strip()

    print("🤖 RAW hallucination output:", text)

    # ✅ Strip ```json or ``` code block markers
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)  # remove ```json or ```text
        text = text.rstrip("`").strip()  # remove trailing backticks

    try:
        unsupported = json.loads(text)

        if not isinstance(unsupported, list):
            raise ValueError("Expected a list")

        # Remove empty/trivial entries
        unsupported = [s for s in unsupported if isinstance(s, str) and s.strip()]
        logger.info(f"Found {len(unsupported)} unsupported.")
        for uns in unsupported:
            print("❌", uns)
        return unsupported

    except Exception as e:
        print("❗ Parsing failed. Falling back to warning mode.")
        print("⚠️ Reason:", str(e))
        return []