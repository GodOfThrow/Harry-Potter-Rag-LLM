"""
llm_factory.py
--------------
สร้าง LLM object ตาม LLM_PROVIDER ใน .env
รองรับ: OpenAI, Azure OpenAI (auto-detect), Google Gemini
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_PROVIDER,
    OPENAI_ENDPOINT,
    OPENAI_MODEL_NAME,
    OPENAI_DEPLOYMENT,
    OPENAI_SUBSCRIPTION_KEY,
    GOOGLE_API_KEY,
    GOOGLE_MODEL_NAME,
)


def get_llm(temperature: float = 0):
    """
    คืน LangChain LLM object ตาม LLM_PROVIDER ที่ตั้งใน .env

    OpenAI ปกติ:  LLM_PROVIDER=openai, เว้น OPENAI_ENDPOINT ว่าง
    Azure OpenAI: LLM_PROVIDER=openai, ใส่ OPENAI_ENDPOINT
    Google Gemini: LLM_PROVIDER=google
    """
    provider = LLM_PROVIDER.lower().strip()

    # ── OpenAI / Azure OpenAI ────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        if not OPENAI_SUBSCRIPTION_KEY:
            raise ValueError(
                "OPENAI_SUBSCRIPTION_KEY ยังไม่ได้ตั้งค่าใน .env\n"
                "ใส่ key แล้วลองใหม่อีกครั้ง"
            )

        # Azure OpenAI — ใช้ ChatOpenAI แบบ v1 API (สำหรับ GPT-5)
        if OPENAI_ENDPOINT:
            print(f"   [LLM] Azure OpenAI (v1 API) | deployment: {OPENAI_DEPLOYMENT}")
            # แปลง Endpoint ให้เป็นรูปแบบ v1
            base_url = OPENAI_ENDPOINT.strip().rstrip('"').rstrip("'").rstrip('/')
            v1_endpoint = f"{base_url}/openai/v1"
            
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=OPENAI_DEPLOYMENT,  # ใช้ชื่อ Deployment แทน Model
                api_key=OPENAI_SUBSCRIPTION_KEY.strip(),
                base_url=v1_endpoint,
                temperature=temperature,
                default_headers={"api-key": OPENAI_SUBSCRIPTION_KEY.strip()}
            )

        # OpenAI ปกติ
        print(f"   [LLM] OpenAI | model: {OPENAI_MODEL_NAME}")
        return ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_SUBSCRIPTION_KEY,
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

        print(f"   [LLM] Google Gemini | model: {GOOGLE_MODEL_NAME}")
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
