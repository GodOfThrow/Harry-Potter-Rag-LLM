"""
build_index.py
--------------
Script สำหรับสร้าง FAISS vector index จากไฟล์ harrypotter.txt
รัน: python utils/build_index.py
"""

import os
import sys

# Fix Windows terminal UTF-8 encoding (for emoji/unicode output)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    SOURCE_TEXT_PATH,
)


def build_index():
    # ── Step 1: Read source text ────────────────────────────────────────────
    if not os.path.exists(SOURCE_TEXT_PATH):
        print(f"❌ Source file not found: {SOURCE_TEXT_PATH}")
        sys.exit(1)

    print(f"📖 Reading: {SOURCE_TEXT_PATH}")
    with open(SOURCE_TEXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"   Total characters: {len(text):,}")

    # ── Step 2: Split into chunks ────────────────────────────────────────────
    print(f"\n✂️  Splitting text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"   Created {len(chunks):,} chunks")

    # ── Step 3: Load embedding model ─────────────────────────────────────────
    print(f"\n🔢 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First run will download ~80MB — please wait...)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Step 4: Build & save FAISS index ─────────────────────────────────────
    print(f"\n🗂️  Building FAISS index from {len(chunks):,} chunks...")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
    vectorstore.save_local(KNOWLEDGE_BASE_PATH)

    print(f"\n✅ Index saved to: {KNOWLEDGE_BASE_PATH}")
    print("   Files: index.faiss, index.pkl")


if __name__ == "__main__":
    build_index()
