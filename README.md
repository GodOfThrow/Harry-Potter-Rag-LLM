# 🧙 Harry Potter RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system built with **LangChain** + **LangGraph** + **Gemini**, using the Harry Potter novel as a knowledge base.

## 🏗️ Architecture

```
User Query (CLI)
      │
      ▼
┌─────────────────────────────┐
│   Agent 1: Data Retriever   │  ← LangGraph ReAct Agent
│   searches knowledge base   │  ← Tool: FAISS Semantic Search
│   via Semantic Search        │       (sentence-transformers)
└──────────────┬──────────────┘
               │ relevant passages
               ▼
┌─────────────────────────────┐
│  Agent 2: Report Generator  │  ← LangChain LCEL Chain
│  synthesizes final answer   │  ← No tools, LLM only
└──────────────┬──────────────┘
               │
               ▼
         Final Answer (CLI)
```

## 📁 Project Structure

```
├── main.py                  # CLI entry point
├── config.py                # Settings (model, chunk size, etc.)
├── harrypotter.txt          # Source novel text
│
├── agents/
│   ├── data_retriever.py    # Agent 1: Semantic search agent
│   └── report_generator.py  # Agent 2: Answer synthesizer
│
├── tools/
│   └── search_tool.py       # FAISS semantic search tool
│
└── utils/
    └── build_index.py       # Script to build FAISS index
```

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API Key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get your key at: https://aistudio.google.com/app/apikey

### 3. Build the knowledge base index
```bash
python utils/build_index.py
```
> First run will download the embedding model (~80MB). This only needs to be done once.

### 4. Run
```bash
python main.py
```

## 💬 Example Usage

```
You: Who is Harry Potter?
You: What is Hogwarts?
You: Describe the relationship between Harry and Voldemort
You: quit
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Agent Framework | LangGraph (ReAct) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| LLM Framework | LangChain |
| Interface | CLI |

## 📝 Notes

- The `knowledge_base/` directory (FAISS index) is excluded from git — run `build_index.py` to regenerate it.
- The `.env` file is excluded from git — never commit API keys.
- If you encounter a `503 UNAVAILABLE` error, the system will automatically retry up to 3 times.
