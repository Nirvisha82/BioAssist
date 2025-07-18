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