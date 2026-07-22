"""
search_tool.py
--------------
Custom LangChain Tool สำหรับ semantic search ใน Harry Potter knowledge base
ใช้ FAISS + HuggingFace embeddings
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, KNOWLEDGE_BASE_PATH, TOP_K

# Lazy-load: โหลด vectorstore เพียงครั้งเดียวแล้ว cache ไว้
_vectorstore = None
_embeddings = None


def _get_vectorstore() -> FAISS:
    """Load FAISS vectorstore (cached after first load)."""
    global _vectorstore, _embeddings

    if _vectorstore is None:
        index_path = os.path.join(KNOWLEDGE_BASE_PATH, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{KNOWLEDGE_BASE_PATH}'.\n"
                "Please run: python utils/build_index.py"
            )

        # print("   [Tool] Loading FAISS index from disk...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            KNOWLEDGE_BASE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        # print("   [Tool] FAISS index loaded ✅")

    return _vectorstore


def semantic_search(query: str) -> str:
    """
    ค้นหา passages ที่เกี่ยวข้องกับ query จาก Harry Potter knowledge base
    คืนค่า: text passages ที่เกี่ยวข้องที่สุด top-k ชุด
    """
    vectorstore = _get_vectorstore()
    docs = vectorstore.similarity_search(query, k=TOP_K)

    if not docs:
        return "No relevant passages found."

    passages = []
    for i, doc in enumerate(docs, start=1):
        passages.append(f"[Passage {i}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(passages)


# LangChain Tool object ที่ Data Retriever Agent จะใช้
harry_potter_search_tool = Tool(
    name="harry_potter_search",
    func=semantic_search,
    description=(
        "Searches the Harry Potter book knowledge base using semantic similarity search. "
        "Use this tool to retrieve relevant text passages that answer questions about "
        "characters, events, places, or any topic from the Harry Potter story. "
        "Input: a natural language search query string."
    ),
)
