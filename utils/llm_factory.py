"""
llm_factory.py
--------------
LLM Factory — สลับ provider ได้แค่เปลี่ยน .env
รองรับ: Google Gemini, OpenAI, OpenAI-compatible (Groq/Together/etc.), Anthropic, Ollama
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL,
)


def get_llm(temperature: float = 0):
    """
    Factory function — คืน LangChain LLM object ตาม LLM_PROVIDER ใน .env

    ตัวอย่างการใช้:
        llm = get_llm(temperature=0)        # for retriever (deterministic)
        llm = get_llm(temperature=0.3)      # for generator (slightly creative)
    """
    provider = LLM_PROVIDER.lower().strip()

    # ── Google Gemini ────────────────────────────────────────────────────────
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY ไม่ได้ตั้งค่าใน .env")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    # ── OpenAI (official) ────────────────────────────────────────────────────
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY ไม่ได้ตั้งค่าใน .env")
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    # ── OpenAI-Compatible endpoint (Groq, Together AI, OpenRouter, LM Studio ฯลฯ) ──
    # ใช้ OpenAI SDK แต่เปลี่ยน base_url → ทำงานกับ API ที่ compatible กับ OpenAI
    elif provider == "openai-compatible":
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY ไม่ได้ตั้งค่าใน .env (ใส่ key ของ provider นั้น)")
        if not OPENAI_BASE_URL:
            raise ValueError("OPENAI_BASE_URL ไม่ได้ตั้งค่าใน .env (เช่น https://api.groq.com/openai/v1)")
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=temperature,
        )

    # ── Anthropic Claude ─────────────────────────────────────────────────────
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY ไม่ได้ตั้งค่าใน .env")
        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    # ── Ollama (Local) ───────────────────────────────────────────────────────
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL or "http://localhost:11434",
            temperature=temperature,
        )

    else:
        supported = ["google", "openai", "openai-compatible", "anthropic", "ollama"]
        raise ValueError(
            f"LLM_PROVIDER '{provider}' ไม่รองรับ\n"
            f"ค่าที่ใช้ได้: {supported}"
        )
