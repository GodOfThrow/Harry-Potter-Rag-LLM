"""
report_generator.py
-------------------
Agent 2: Report Generator
Role: Expert writer and synthesizer
Tool: None (LLM only)
Input: user query + context passages from Data Retriever
Output: Final comprehensive, well-written answer for the user
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, GOOGLE_API_KEY

# ── System Prompt ────────────────────────────────────────────────────────────
GENERATOR_SYSTEM_PROMPT = """You are an expert writer and literary analyst specializing in the Harry Potter universe.

Your task is to synthesize the provided text passages into a comprehensive, accurate, and engaging answer for the user.

RULES:
1. Base your answer ONLY on the provided passages — do not invent details.
2. If passages contain insufficient information, clearly state what you found and what is missing.
3. Write in a clear, well-structured, and engaging style.
4. Use paragraph breaks for readability.
5. If quoting directly from the text, use quotation marks.

---
CONTEXT PASSAGES (retrieved from Harry Potter book):
{context}

---
USER QUESTION:
{query}

---
YOUR COMPREHENSIVE ANSWER:"""


def create_report_generator():
    """สร้าง Report Generator chain (LCEL)"""

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,        # เล็กน้อย creative เพื่อการเขียนที่ดี
    )

    prompt = ChatPromptTemplate.from_template(GENERATOR_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain


# Singleton instance
report_generator = create_report_generator()
