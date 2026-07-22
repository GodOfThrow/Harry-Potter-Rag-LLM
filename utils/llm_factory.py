"""
llm_factory.py
--------------
สร้าง LLM object ตาม LLM_PROVIDER ใน .env
รองรับ: OpenAI, Azure OpenAI, Google Gemini
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_PROVIDER,
    OPENAI_ENDPOINT,
    OPENAI_MODEL_NAME,
    OPENAI_DEPLOYMENT,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_MODEL_NAME,
)


def get_llm(temperature: float = 0):
    """
    คืน LangChain LLM object ตาม LLM_PROVIDER ที่ตั้งใน .env

    OpenAI (ปกติ):  ตั้ง LLM_PROVIDER=openai, เว้น OPENAI_ENDPOINT ว่าง
    Azure OpenAI:   ตั้ง LLM_PROVIDER=openai, ใส่ OPENAI_ENDPOINT
    Google Gemini:  ตั้ง LLM_PROVIDER=google
    """
    provider = LLM_PROVIDER.lower().strip()

    # ── OpenAI / Azure OpenAI ────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY ยังไม่ได้ตั้งค่าใน .env\n"
                "ใส่ key แล้วลองใหม่อีกครั้ง"
            )

        # Azure OpenAI — ถ้ามี endpoint ให้ใช้ AzureChatOpenAI
        if OPENAI_ENDPOINT:
            print(f"   [LLM] Using Azure OpenAI | deployment: {OPENAI_DEPLOYMENT}")
            return AzureChatOpenAI(
                azure_endpoint=OPENAI_ENDPOINT,
                azure_deployment=OPENAI_DEPLOYMENT,
                api_key=OPENAI_API_KEY,
                api_version="2024-08-01-preview",
                temperature=temperature,
            )

        # OpenAI ปกติ — ไม่มี endpoint
        print(f"   [LLM] Using OpenAI | model: {OPENAI_MODEL_NAME}")
        return ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    # ── Google Gemini ────────────────────────────────────────────────────────
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY ยังไม่ได้ตั้งค่าใน .env\n"
                "รับ key ได้ที่: https://aistudio.google.com/app/apikey"
            )

        print(f"   [LLM] Using Google Gemini | model: {GOOGLE_MODEL_NAME}")
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' ไม่รองรับ\n"
            f"ค่าที่ใช้ได้: 'openai' หรือ 'google'"
        )
