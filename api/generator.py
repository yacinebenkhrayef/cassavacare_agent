import os
from google import genai
from google.genai import types
from api.models import SourceChunk
from typing import List

# The SDK automatically looks for the GEMINI_API_KEY environment variable.
# For the free tier, use "gemini-2.5-flash" (excellent for speed and RAG tasks)
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are CassavaCare, an expert agronomic assistant specialising in cassava leaf disease diagnosis and treatment.

Rules:
- Answer ONLY using the context passages provided below.
- If the context does not contain enough information to answer, say: "I could not find sufficient information in the available documents."
- Always be specific: mention disease names, symptoms, and treatments as they appear in the context.
- Keep your answer concise (3-5 sentences) unless the question requires more detail.
- Do not invent facts or cite sources not present in the context."""

def build_prompt(question: str, chunks: List[SourceChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i} — {chunk.source}]\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"

def generate_answer(question: str, chunks: List[SourceChunk]) -> str:
    if not chunks:
        return "No relevant documents were found for your question."

    user_prompt = build_prompt(question, chunks)

    # Configure system instructions and hyperparameters
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,       # Low temperature for factual RAG answers
        max_output_tokens=512,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=config,
    )
    
    return response.text.strip()