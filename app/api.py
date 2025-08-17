from fastapi import FastAPI
from pydantic import BaseModel
from .graph import APP

app = FastAPI(title="Mini RAG (LangGraph)")

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: Ask):
    state = {"question": payload.question, "docs": [], "answer": ""}
    out = APP.invoke(state)
    return {"answer": out["answer"]}
