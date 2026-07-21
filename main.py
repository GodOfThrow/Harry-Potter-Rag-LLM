"""
main.py
-------
Harry Potter RAG System — CLI Entry Point

รัน: python main.py
"""

import os
import sys

# Fix Windows terminal UTF-8 encoding (for emoji/unicode output)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

# โหลด .env ก่อนทุกอย่าง
load_dotenv()

# ตรวจ API Key ก่อนเริ่ม
from config import GOOGLE_API_KEY, KNOWLEDGE_BASE_PATH

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🧙  Harry Potter RAG System  🧙                  ║
║   Powered by LangChain + Gemini + Semantic Search        ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  <question>   — Ask anything about Harry Potter
  help         — Show this help message
  quit / exit  — Exit the program
"""


def check_api_key():
    """ตรวจสอบว่ามี API Key หรือไม่"""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        print("❌ ERROR: GOOGLE_API_KEY not set!")
        print("   Please edit the .env file and add your Google API key:")
        print("   GOOGLE_API_KEY=your_actual_key_here")
        sys.exit(1)


def check_and_build_index():
    """ตรวจว่า FAISS index มีอยู่แล้ว ถ้าไม่มีให้ build อัตโนมัติ"""
    index_file = os.path.join(KNOWLEDGE_BASE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        print("⚠️  Knowledge base index not found.")
        print("🔨 Building index automatically (first-time setup)...\n")
        from utils.build_index import build_index
        build_index()
        print()


def run_pipeline(query: str) -> str:
    """
    รัน RAG pipeline:
    1. Data Retriever Agent → ดึง relevant passages
    2. Report Generator Agent → สังเคราะห์คำตอบ
    มี retry logic สำหรับ 503 UNAVAILABLE จาก Gemini API
    """
    import time
    from config import MAX_RETRIES, RETRY_DELAY

    # Import ที่นี่เพื่อหลีกเลี่ยงการโหลด model ก่อนตรวจ API key
    from agents.data_retriever import data_retriever
    from agents.report_generator import report_generator

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ── Agent 1: Data Retriever ──────────────────────────────────────
            print("\n" + "─" * 60)
            print("📚  [Agent 1] Data Retriever — Searching knowledge base...")
            print("─" * 60)

            retriever_result = data_retriever.invoke({"input": query})
            context = retriever_result.get("output", "No relevant passages found.")

            # ── Agent 2: Report Generator ────────────────────────────────────
            print("\n" + "─" * 60)
            print("✍️   [Agent 2] Report Generator — Synthesizing answer...")
            print("─" * 60)

            final_answer = report_generator.invoke({"query": query, "context": context})
            return final_answer

        except Exception as e:
            err_str = str(e)
            # ตรวจว่าเป็น 503 หรือ rate limit error
            if any(code in err_str for code in ["503", "429", "UNAVAILABLE", "Resource has been exhausted"]):
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt   # exponential: 5s, 10s, 15s
                    print(f"\n⚠️  API busy (attempt {attempt}/{MAX_RETRIES}). Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return (
                        f"❌ API ไม่ตอบสนองหลังจากลอง {MAX_RETRIES} ครั้ง\n"
                        f"   กรุณารอสักครู่แล้วลองใหม่ หรือเปลี่ยน LLM_MODEL ใน config.py\n"
                        f"   Error: {err_str[:200]}"
                    )
            else:
                raise  # error อื่น → throw ปกติ

    return final_answer


def main():
    print(BANNER)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    check_api_key()
    check_and_build_index()

    print("✅ System ready!\n")
    print(HELP_TEXT)

    # ── CLI Loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            if query.lower() == "help":
                print(HELP_TEXT)
                continue

            # Run pipeline
            answer = run_pipeline(query)

            # Display final answer
            print("\n" + "═" * 60)
            print("🤖  Final Answer:")
            print("═" * 60)
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
