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

```text
You: How many presents did Dudley Dursley receive on his birthday?

────────────────────────────────────────────────────────────
📚  [Agent 1] Data Retriever — Searching knowledge base...
────────────────────────────────────────────────────────────

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
On his birthday, Dudley Dursley received thirty-seven presents. He initially counted thirty-six, but his mother reminded him that he hadn't included "Auntie Marge's present," which brought the total to thirty-seven. The table was "almost hidden beneath all Dudley's birthday presents," which included a new computer, a second television, and a racing bike.

You: Why did Harry and Ron become friends with Hermione after Halloween?

────────────────────────────────────────────────────────────
📚  [Agent 1] Data Retriever — Searching knowledge base...
────────────────────────────────────────────────────────────

   [Retriever] Searching for: 'Why did Harry and Ron become friends with Hermione after Halloween?'
   [Retriever] Retrieved 1 characters of context

────────────────────────────────────────────────────────────
✍️   [Agent 2] Report Generator — Synthesizing answer...
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
Harry and Ron became friends with Hermione after Halloween because they "saved her from the mountain troll."

Before this event, their relationship was strained. Ron openly expressed his dislike for Hermione, calling her a "nightmare" and stating that "no one can stand her" (Passage 5). Hermione, described as a "bossy know-it-all," had been refusing to speak to them (Passage 4, Passage 2). Ron's harsh comments led Hermione to be found crying in the girls' bathroom on Halloween (Passage 5, Passage 1).

However, after Harry and Ron saved her from the mountain troll, Hermione's demeanor changed significantly. She "had become a bit more relaxed about breaking rules... and she was much nicer for it" (Passage 3). This pivotal event transformed their dynamic, leading to the formation of their friendship. The passages indicate that this change occurred around Halloween, as the positive shift in Hermione's behavior is noted before Harry's first Quidditch match, which would have taken place after the Halloween feast.

You: What was the first password to the Gryffindor common room?

────────────────────────────────────────────────────────────
📚  [Agent 1] Data Retriever — Searching knowledge base...
────────────────────────────────────────────────────────────

   [Retriever] Searching for: 'What was the first password to the Gryffindor common room?'
   [Retriever] Retrieved 1 characters of context

────────────────────────────────────────────────────────────
✍️   [Agent 2] Report Generator — Synthesizing answer...
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
Based on the provided text, the first password to the Gryffindor common room was "Caput Draconis." Percy Weasley provided this password, causing the portrait to swing open and reveal the entrance to the common room.
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | LangGraph (ReAct) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| LLM Framework | LangChain |
| Interface | CLI |

## 📝 Notes

- The `knowledge_base/` directory (FAISS index) is excluded from git — run `build_index.py` to regenerate it.
- The `.env` file is excluded from git — never commit API keys.
- If you encounter a `503 UNAVAILABLE` error, the system will automatically retry up to 3 times.
