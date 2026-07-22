"""
llm_factory.py
--------------
สร้าง LLM object ตาม LLM_PROVIDER ใน .env
รองรับ: OpenAI, Azure OpenAI, Azure APIM, Google Gemini
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

    - openai + ENDPOINT (.openai.azure.com) → AzureChatOpenAI
    - openai + ENDPOINT (APIM/อื่นๆ)        → ChatOpenAI + base_url
    - openai + ไม่มี ENDPOINT              → ChatOpenAI (OpenAI ปกติ)
    - google                               → ChatGoogleGenerativeAI
    """
    provider = LLM_PROVIDER.lower().strip()

    # ── OpenAI / Azure / APIM ────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_SUBSCRIPTION_KEY:
            raise ValueError(
                "OPENAI_SUBSCRIPTION_KEY ยังไม่ได้ตั้งค่าใน .env\n"
                "ใส่ key แล้วลองใหม่อีกครั้ง"
            )

        if OPENAI_ENDPOINT:
            endpoint = OPENAI_ENDPOINT.rstrip("/")

            # ── Standard Azure OpenAI (.openai.azure.com) ────────────────────
            if ".openai.azure.com" in endpoint:
                from langchain_openai import AzureChatOpenAI
                # print(f"   [LLM] Azure OpenAI | deployment: {OPENAI_DEPLOYMENT}")
                return AzureChatOpenAI(
                    azure_endpoint=endpoint,
                    azure_deployment=OPENAI_DEPLOYMENT,
                    api_key=OPENAI_SUBSCRIPTION_KEY,
                    api_version="2024-08-01-preview",
                    temperature=temperature,
                )

            # ── Azure APIM / Custom Endpoint → ใช้ /chat/completions ─────────
            else:
                import httpx

                base_url = endpoint
                for suffix in ["/responses", "/chat/completions", "/completions", "/v1"]:
                    if base_url.endswith(suffix):
                        base_url = base_url[: -len(suffix)]
                        break

                # ใช้ event hook — inject api-key header ทุก request
                _key = OPENAI_SUBSCRIPTION_KEY
                def _inject_apim_key(request: httpx.Request) -> None:
                    request.headers["api-key"] = _key

                http_client = httpx.Client(
                    event_hooks={"request": [_inject_apim_key]}
                )

                # print(f"   [LLM] Azure APIM | base_url: {base_url} | model: {OPENAI_DEPLOYMENT}")
                return ChatOpenAI(
                    model=OPENAI_DEPLOYMENT,
                    api_key=OPENAI_SUBSCRIPTION_KEY,
                    base_url=base_url,
                    temperature=temperature,
                    http_client=http_client,
                )

        # ── OpenAI ปกติ (ไม่มี endpoint) ────────────────────────────────────
        # print(f"   [LLM] OpenAI | model: {OPENAI_MODEL_NAME}")
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

        # print(f"   [LLM] Google Gemini | model: {GOOGLE_MODEL_NAME}")
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
