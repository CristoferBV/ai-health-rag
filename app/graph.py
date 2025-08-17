from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from .retriever import build_retriever
from .prompts import SYSTEM_PROMPT
import os

class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str

retriever = build_retriever(k=5)
llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini"), temperature=0)

def node_retrieve(state: State) -> dict:
    docs = retriever.get_relevant_documents(state["question"])
    return {"docs": docs}

# Rerank opcional: aquí puedes reordenar por score/longitud/etc.
def node_rerank(state: State) -> dict:
    docs = state["docs"]
    return {"docs": docs[:5]}

def node_answer(state: State) -> dict:
    context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(state["docs"], 1)])
    citations = " ".join([f"[{i}:{d.metadata.get('source','') or d.metadata.get('file_path','')}]" 
                          for i, d in enumerate(state["docs"], 1)])
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQ: {state['question']}\nA:"
    resp = llm.invoke(prompt)
    return {"answer": f"{resp.content}\n\nCitations: {citations}"}

graph = StateGraph(State)
graph.add_node("retrieve", node_retrieve)
graph.add_node("rerank", node_rerank)
graph.add_node("answer", node_answer)
graph.add_edge("retrieve","rerank")
graph.add_edge("rerank","answer")
graph.set_entry_point("retrieve")
graph.add_edge("answer", END)
APP = graph.compile()
