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

#### 🔴 Google Gemini (default)
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL_NAME=gemini-2.5-flash
```
> Get your Gemini key at: https://aistudio.google.com/app/apikey

#### 🔷 OpenAI
```env
LLM_PROVIDER=openai
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_MODEL_NAME=gpt-5-mini
OPENAI_DEPLOYMENT=gpt-5-mini
OPENAI_SUBSCRIPTION_KEY=your_azure_subscription_key
```

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

### 🔴 Google Gemini 2.5 Flash

```text
You: How many presents did Dudley Dursley receive on his birthday?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
On his birthday, Dudley Dursley initially counted thirty-six presents. However, his mother then reminded him about "Auntie Marge's present," which was hidden "under this big one from Mommy and Daddy." After this was pointed out, Dudley acknowledged the additional gift, stating, "All right, thirty-seven then." Therefore, Dudley received a total of thirty-seven presents.

You: Why did Harry and Ron become friends with Hermione after Halloween?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
On Halloween, Hermione was deeply upset and crying in the girls' bathroom after overhearing Ron's harsh comments, where he called her a "nightmare" and suggested "she's got no friends" (Passages 1, 5). Harry even caught a glimpse of her face and was "startled to see that she was in tears" (Passage 5).

However, Harry and Ron became friends with Hermione after this initial period of animosity. The provided passages indicate that this significant shift in their relationship occurred "since Harry and Ron had saved her from the mountain troll" (Passage 3). Following this event, Hermione's demeanor changed considerably; she "had become a bit more relaxed about breaking rules... and she was much nicer for it" (Passage 3). An example of her changed behavior and their newfound camaraderie is her conjuring a bright blue, portable fire for them in a jam jar during a break (Passage 3).

The provided passages do not explicitly detail the mountain troll incident itself or state its precise timing relative to the Halloween feast, only that the friendship developed *since* Harry and Ron saved her from the troll.

You: What was the first password to the Gryffindor common room?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
The first password to the Gryffindor common room, as stated in the provided text, was "Caput Draconis." Percy Weasley provided this password, causing the portrait to swing open and reveal the entrance to the common room.
```

### 🔵 OpenAI (gpt-5-mini)

```text
You: How many presents did Dudley Dursley receive on his birthday?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
Dudley received thirty-seven presents. In the passage he first says "Thirty-six," then, after his mother points out Auntie Marge's gift, he corrects himself: "All right, thirty-seven then."

You: Why did Harry and Ron become friends with Hermione after Halloween?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
They became friends because of the shared, life‑threatening experience of saving Hermione from the mountain troll on Halloween. As the text puts it, "But from that moment on, Hermione Granger became their friend. There are some things you can't share without ending up liking each other, and knocking out a twelve-foot mountain troll is one of them." The three of them had to work together under extreme danger (Ron casting "Wingardium Leviosa" to disarm the troll, Harry distracting and holding on to it), and that adrenaline‑filled, loyal rescue forged an immediate bond.

The passages also show Hermione defending Harry and Ron afterward — she hung her head and even pretended she had tackled the troll to get them out of trouble — which likely strengthened their mutual trust. Finally, the books note Hermione "had become a bit more relaxed about breaking rules since Harry and Ron had saved her... and she was much nicer for it," signaling a change in their relationship after the incident.

If you want more about how their friendship developed later, the provided passages stop here and don’t give further detail.

You: What was the first password to the Gryffindor common room?

════════════════════════════════════════════════════════════
🤖  Final Answer:
════════════════════════════════════════════════════════════
The passage shows the password being spoken aloud: when asked "Password?" Percy answers, "Caput Draconis." The portrait then swings forward and they enter the Gryffindor common room.
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (default) | Google Gemini 2.5 Flash |
| LLM (alternative) | OpenAI GPT-5-mini |
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
