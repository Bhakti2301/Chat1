from __future__ import annotations

from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
import os
import time
import sqlite3
import sys

try:
    # Optional: only used if OPENAI_API_KEY is set
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


DB_PATH = os.path.join(os.path.dirname(__file__), "convo.db")


class ChatState(TypedDict, total=False):
    step: str
    session_id: str
    message: str
    response: str
    error: Optional[str]


def ensure_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        conn.commit()


def save_message(session_id: str, role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        conn.commit()


def fetch_recent_messages(session_id: str, limit: int = 12) -> List[Dict[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    # reverse to chronological
    rows = rows[::-1]
    return [{"role": r, "content": c} for (r, c) in rows]


def generate_with_llm(messages: List[Dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            # Use a small, widely available model name; adjust if needed
            completion = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.4,
                max_tokens=300,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            return f"[LLM error] {exc}"

    # Fallback deterministic template if no key
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    context = " ".join(m["content"] for m in messages if m["role"] == "assistant")
    prefix = "(LLM unavailable) " if not api_key else ""
    return f"{prefix}You said: '{last_user}'. I remember: '{context[-200:]}'"


# Nodes
def start_node(state: ChatState) -> ChatState:
    ensure_db()
    return {"step": "RETRIEVE"}


def retrieve_node(state: ChatState) -> ChatState:
    session_id = state.get("session_id") or "default"
    recent = fetch_recent_messages(session_id, limit=12)
    state["_recent"] = recent  # type: ignore
    return {"step": "LLM"}


def llm_node(state: ChatState) -> ChatState:
    session_id = state.get("session_id") or "default"
    user_text = (state.get("message") or "").strip()
    # Build messages with a simple system prompt
    base: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a helpful, concise assistant. Use prior conversation context to keep continuity.",
        }
    ]
    recent: List[Dict[str, str]] = state.get("_recent", [])  # type: ignore
    compiled: List[Dict[str, str]] = base + recent + [{"role": "user", "content": user_text}]

    reply = generate_with_llm(compiled)
    # Save to DB
    save_message(session_id, "user", user_text)
    save_message(session_id, "assistant", reply)
    return {"response": reply, "step": "RESPOND"}


def respond_node(state: ChatState) -> ChatState:
    return {"step": "END"}


# Build graph
graph = StateGraph(ChatState)
graph.add_node("START", start_node)
graph.add_node("RETRIEVE", retrieve_node)
graph.add_node("LLM", llm_node)
graph.add_node("RESPOND", respond_node)

graph.set_entry_point("START")
graph.add_edge("START", "RETRIEVE")
graph.add_edge("RETRIEVE", "LLM")
graph.add_edge("LLM", "RESPOND")
graph.add_edge("RESPOND", END)

app = graph.compile()


def main() -> None:
    # One-shot: args => session_id optional via CHAT_SESSION_ID env
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        session_id = os.environ.get("CHAT_SESSION_ID", "default")
        state: ChatState = {"step": "START", "session_id": session_id, "message": message}
        state = app.invoke(state)
        print(state.get("response", state.get("error", "(no output)")))
        return

    print("LLM + SQLite Memory Chatbot")
    print("=" * 40)
    print("Set OPENAI_API_KEY for real LLM replies. Type 'exit' to quit.")
    session_id = os.environ.get("CHAT_SESSION_ID", "default")
    try:
        while True:
            user_in = input("You: ")
            if user_in.strip().lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye!")
                break
            state: ChatState = {"step": "START", "session_id": session_id, "message": user_in}
            state = app.invoke(state)
            print("Bot:", state.get("response", state.get("error", "(no output)")))
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")


if __name__ == "__main__":
    main()


