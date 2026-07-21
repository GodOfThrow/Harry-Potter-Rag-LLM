import os
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_TEXT_PATH = os.path.join(BASE_DIR, "harrypotter.txt")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base")

# ---- Chunking ----
CHUNK_SIZE = 500        # จำนวน characters ต่อ chunk
CHUNK_OVERLAP = 50      # overlap ระหว่าง chunks

# ---- Search ----
TOP_K = 5               # จำนวน chunks ที่ดึงมาจาก FAISS

# ---- Models ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local model (~80MB)
LLM_MODEL = "gemini-2.5-flash"

# ---- Retry ----
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retries

# ---- API Keys ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
