# 🧙 Harry Potter RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system built with **LangChain** + **LangGraph** + **FAISS Semantic Search**, using the Harry Potter novel as a knowledge base. Supports **OpenAI**, **Azure OpenAI**, and **Google Gemini** — switchable via `.env` without touching any code.

## 🏗️ Architecture

```
User Query (CLI)
      │
      ▼
┌─────────────────────────────────────┐
│      Agent 1: Data Retriever        │  ← LangGraph ReAct Agent
│  Searches knowledge base via        │  ← Tool: FAISS Semantic Search
│  semantic similarity search         │       (sentence-transformers)
└──────────────────┬──────────────────┘
                   │ relevant passages
                   ▼
┌─────────────────────────────────────┐
│     Agent 2: Report Generator       │  ← LangChain LCEL Chain
│  Synthesizes a comprehensive        │  ← No tools, LLM only
│  answer from retrieved passages     │
└──────────────────┬──────────────────┘
                   │
                   ▼
            Final Answer (CLI)
```

## 📁 Project Structure

```
├── main.py                    # CLI entry point
├── config.py                  # All settings (models, paths, provider)
├── .env                       # API keys & provider selection (not in git)
├── harrypotter.txt            # Source novel text
├── requirements.txt
│
├── agents/
│   ├── data_retriever.py      # Agent 1: LangGraph ReAct search agent
│   └── report_generator.py   # Agent 2: LCEL answer synthesizer
│
├── tools/
│   └── search_tool.py         # Custom LangChain Tool (FAISS semantic search)
│
└── utils/
    ├── build_index.py         # Script to build FAISS vector index
    └── llm_factory.py         # LLM factory — returns correct LLM per provider
```

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file in the project root and choose your provider:

#### 🔵 OpenAI (default)
```env
LLM_PROVIDER=openai
OPENAI_ENDPOINT=
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_DEPLOYMENT=gpt-4o-mini
OPENAI_SUBSCRIPTION_KEY=sk-xxxxxxxxxxxxxxxx
```

#### 🔷 Azure OpenAI
```env
LLM_PROVIDER=openai
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_DEPLOYMENT=gpt-4o-mini
OPENAI_SUBSCRIPTION_KEY=your_azure_subscription_key
```

#### 🔴 Google Gemini
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL_NAME=gemini-2.0-flash
```
> Get your Gemini key at: https://aistudio.google.com/app/apikey

### 3. Build the knowledge base index
```bash
python utils/build_index.py
```
> First run will download the embedding model (~80MB). Only needs to be done **once**.

### 4. Run
```bash
python main.py
```

## 🔄 Switching LLM Provider

Only two steps — **no code changes needed**:

1. Edit `LLM_PROVIDER` in `.env` → `openai` or `google`
2. Fill in the matching API key
3. Run `python main.py`

The system auto-detects **Azure vs regular OpenAI** based on whether `OPENAI_ENDPOINT` is set.

## 💬 Example Output

```text
You: How many presents did Dudley Dursley receive on his birthday?

────────────────────────────────────────────────────────────
📚  [Agent 1] Data Retriever — Searching knowledge base...
────────────────────────────────────────────────────────────

   [LLM] OpenAI | model: gpt-4o-mini
   [Retriever] Searching for: 'How many presents did Dudley Dursley receive on his birthday?'
   [Tool] Loading FAISS index from disk...
   [Tool] FAISS index loaded ✅
   [Retriever] Retrieved 1 characters of context

────────────────────────────────────────────────────────────
✍️   [Agent 2] Report Generator — Synthesizing answer...
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
On his birthday, Dudley Dursley received thirty-seven presents. He initially
counted thirty-six, but his mother reminded him that he hadn't included
"Auntie Marge's present," which brought the total to thirty-seven. The table
was "almost hidden beneath all Dudley's birthday presents," which included a
new computer, a second television, and a racing bike.

You: Why did Harry and Ron become friends with Hermione after Halloween?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
Harry and Ron became friends with Hermione after Halloween because they
"saved her from the mountain troll."

Before this event, their relationship was strained. Ron openly expressed his
dislike for Hermione, calling her a "nightmare" and stating that "no one can
stand her." Ron's harsh comments led Hermione to be found crying in the girls'
bathroom on Halloween.

However, after Harry and Ron saved her from the mountain troll, Hermione's
demeanor changed significantly. She "had become a bit more relaxed about
breaking rules... and she was much nicer for it."

You: What was the first password to the Gryffindor common room?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
The first password to the Gryffindor common room was "Caput Draconis."
Percy Weasley provided this password, causing the portrait to swing open
and reveal the entrance to the common room.
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (default) | OpenAI GPT-4o-mini |
| LLM (alternative) | Google Gemini 2.0 Flash |
| LLM (alternative) | Azure OpenAI |
| Agent Framework | LangGraph (ReAct) |
| LLM Orchestration | LangChain LCEL |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (local, no server needed) |
| Interface | CLI |

## 📝 Notes

- `knowledge_base/` (FAISS index) is excluded from git — run `build_index.py` to regenerate.
- `.env` is excluded from git — never commit API keys.
- The system auto-retries up to **3 times** on `503 UNAVAILABLE` / `429` rate-limit errors.
- Embedding model runs **locally** — no API key or internet required after first download.
