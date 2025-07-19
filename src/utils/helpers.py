from transformers import pipeline
import re
from loguru import logger

nli = pipeline("text-classification", model="roberta-large-mnli")

def guardrail(gemini_model, query):
    classification_prompt = f"""
You are a binary classifier.  
– If the user’s question is about health care, biomedical science, clinical topics, medicine, or about the assistant itself (e.g. “Who are you?”, “What can you do?”, simple greetings or thank‑yous), answer “Yes”.
- If you get this query : "The user is asking about BioAssist Chatbot about myself." Say Yes.  
– Otherwise, answer “No”.  

Respond with **only** “Yes” or “No”.

Question: "{query}"  
Answer (Yes or No only):
"""
    response = gemini_model.generate_content(classification_prompt)
    return response.text.strip().lower() == "yes"

def rewrite_query(gemini_model, chat_history, query):
    history = "\n".join([f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in chat_history])
    rewrite_prompt = f"""
Given the conversation below and the latest user question, rewrite the question so that it is fully self‑contained and does not use pronouns or ambiguous references.
If the question is asking about the assistant itself (e.g. “Who are you?”, “What can you do?”, simple greetings or thank‑yous):
  Resppond only with: “The user is asking about BioAssist Chatbot about myself.”
This rewritten question will be used for web search. Only provide the rewritten question and nothing else in your response.

Chat:
{history}

Latest Question: "{query}"

Rewritten Question:
"""
    response = gemini_model.generate_content(rewrite_prompt)
    return response.text.strip()



def split_sentences(text):
    # very basic sentence splitter
    return re.split(r'(?<=[.!?])\s+', text.strip())

import json

def verify_grounding(answer: str, contexts: list[str],gemini_model) -> list[str]:
    """
    Ask the LLM itself to flag any sentences in `answer` 
    that aren’t directly supported by the concatenated `contexts`.
    Returns a list of unsupported sentences (or [] if all are fine).
    """
    # 1. Build the prompt
    prompt = f"""
You are a fact‑checker. Here is the provided context (sources):

{'\n\n'.join(contexts)}

Here is the assistant’s answer:

{answer}

Please list, as a JSON array, any sentences from the answer that are NOT fully supported by the context above. 
If every sentence is grounded, return an empty list: [].
"""
    # 2. Call the LLM
    resp = gemini_model.generate_content(prompt)
    text = resp.text.strip()
    print(text)
    # 3. Try to parse out a JSON list
    try:
        unsupported = json.loads(text)
        if not isinstance(unsupported, list):
            raise ValueError
    except Exception:
        # fallback: return the raw text if parsing failed
        return [text]
    
    logger.info(f"Found - {len(unsupported)} unsupported.")
    for uns in unsupported:
        print(uns)
    logger.info("Done.")
    return unsupported