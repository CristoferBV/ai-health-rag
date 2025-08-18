from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from .retriever import build_retriever
from .prompts import SYSTEM_PROMPT

# 🔹 Modo OFFLINE: sin OpenAI, sin API keys.
#    - Usa embeddings locales (ya los creaste en la ingesta)
#    - Responde de forma extractiva (con citas) usando los mejores trozos

class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str

retriever = build_retriever(k=5)

def node_retrieve(state: State) -> dict:
    docs = retriever.get_relevant_documents(state["question"])
    return {"docs": docs}

def node_rerank(state: State) -> dict:
    # Aquí podrías aplicar un rerank propio; por simplicidad, top-5
    return {"docs": state["docs"][:5]}

def _citations(docs: List[Document]) -> str:
    # muestra [idx:source] para cada chunk usado
    return " ".join(
        f"[{i+1}:{(d.metadata.get('source') or d.metadata.get('file_path') or '')}]"
        for i, d in enumerate(docs)
    )

def node_answer(state: State) -> dict:
    docs = state["docs"]
    if not docs:
        return {"answer": "I don't have enough context to answer.\n\nCitations: []"}

    # 🔸 Respuesta extractiva sencilla (sin LLM):
    #     concatenamos pequeños fragmentos de los mejores trozos como evidencia.
    max_chars = 700  # controla longitud de la respuesta
    collected = []
    used = 0
    for d in docs[:2]:  # toma 1–2 trozos top
        txt = d.page_content.strip().replace("\n", " ")
        take = txt[: max(0, max_chars - used)]
        if take:
            collected.append(take)
            used += len(take)
        if used >= max_chars:
            break

    answer = (
        f"(offline) Based on the retrieved context, here is a concise answer:\n"
        + " ".join(collected)
    )
    return {"answer": f"{answer}\n\nCitations: {_citations(docs)}"}

# ➜ Grafo LangGraph
graph = StateGraph(State)
graph.add_node("retrieve", node_retrieve)
graph.add_node("rerank", node_rerank)
graph.add_node("answer", node_answer)
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "answer")
graph.set_entry_point("retrieve")
graph.add_edge("answer", END)
APP = graph.compile()
