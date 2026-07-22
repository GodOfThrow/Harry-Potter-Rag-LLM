"""
data_retriever.py
-----------------
Agent 1: Data Retriever
Role: Expert information retrieval specialist
Tool: harry_potter_search (semantic search via FAISS)
Framework: LangGraph create_react_agent
LLM: ดึงจาก llm_factory → สลับระหว่าง OpenAI / Google ได้จาก .env
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from tools.search_tool import harry_potter_search_tool
from utils.llm_factory import get_llm

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert information retrieval specialist for the Harry Potter series. "
    "Your ONLY job is to use the harry_potter_search tool to find ALL text passages "
    "relevant to the user's query. "
    "IMPORTANT: Always call the tool at least once. "
    "Do NOT summarize or analyze — return the raw retrieved passages as your final answer."
)


def create_data_retriever() -> RunnableLambda:
    """สร้าง Data Retriever Agent ด้วย LangGraph ReAct pattern"""

    llm = get_llm(temperature=0)
    tools = [harry_potter_search_tool]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    def _invoke(inputs: dict) -> dict:
        """Wrapper: invoke({"input": query}) -> {"output": passages}"""
        query = inputs["input"]
        print(f"\n   [Retriever] Searching for: {query!r}")

        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })

        last_message = result["messages"][-1]
        output = last_message.content
        print(f"   [Retriever] Retrieved {len(output)} characters of context")
        return {"output": output}

    return RunnableLambda(_invoke)


# Singleton instance
data_retriever = create_data_retriever()
