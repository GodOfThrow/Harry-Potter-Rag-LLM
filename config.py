import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_TEXT_PATH = os.path.join(BASE_DIR, "harrypotter.txt")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Search ────────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Embedding Model (local, ไม่ต้อง API) ──────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Provider Selection ────────────────────────────────────────────────────────
# "openai"  → OpenAI หรือ Azure OpenAI (default)
# "google"  → Google Gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# ── OpenAI / Azure OpenAI ─────────────────────────────────────────────────────
OPENAI_ENDPOINT    = os.getenv("OPENAI_ENDPOINT", "")       # Azure endpoint URL (ว่าง = OpenAI ปกติ)
OPENAI_MODEL_NAME  = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_DEPLOYMENT  = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o-mini")  # Azure deployment name
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

# ── Google Gemini ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL_NAME  = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")

# ── Retry (สำหรับ 503 / 429) ───────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 5     # seconds (x attempt: 5s, 10s, 15s)
