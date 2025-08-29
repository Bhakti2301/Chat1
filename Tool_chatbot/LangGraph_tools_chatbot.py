from __future__ import annotations

from typing import TypedDict, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import math
import operator
import re
import json
import urllib.parse
import urllib.request
import html as html_lib
import sys


class ChatState(TypedDict, total=False):
    step: str
    message: str
    intent: Literal["search", "calc", "chat"]
    tool_result: str
    error: Optional[str]


def web_search(query: str, num_results: int = 5) -> str:
    """Lightweight web search using DuckDuckGo HTML results.

    Returns a newline-joined list of title - url pairs.
    """
    def fetch_html(base: str) -> str:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"{base}/html/?{encoded}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    # Try privacy HTML endpoint first, then standard HTML as fallback
    html_pages: list[str] = []
    try:
        html_pages.append(fetch_html("https://html.duckduckgo.com"))
    except Exception:
        pass
    try:
        html_pages.append(fetch_html("https://duckduckgo.com"))
    except Exception:
        pass

    results: list[str] = []
    for html in html_pages:
        if results:
            break
        # Multiple patterns seen in DDG HTML
        link_patterns = [
            r"<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
            r"<h2[^>]*class=\"result__title\"[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
            r"<a[^>]+class=\"result__url[^\"]*\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        ]
        for pattern in link_patterns:
            for m in re.finditer(pattern, html, flags=re.I | re.S):
                href = m.group(1)
                title_raw = m.group(2)
                title = html_lib.unescape(re.sub("<.*?>", "", title_raw)).strip()
                if not title:
                    continue
                if href.startswith("/l/?"):
                    # DuckDuckGo redirect links, extract 'uddg'
                    parsed = urllib.parse.urlparse("https://duckduckgo.com" + href)
                    qs = urllib.parse.parse_qs(parsed.query)
                    real = qs.get("uddg", [href])[0]
                    href = urllib.parse.unquote(real)
                if href.startswith("//"):
                    href = "https:" + href
                href = html_lib.unescape(href)
                results.append(f"{title} - {href}")
                if len(results) >= num_results:
                    break
            if results:
                break

    if not results:
        return "No results found."
    return "\n".join(results)


ALLOWED_OPERATORS: Dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}


def safe_calculate(expression: str) -> str:
    """Evaluate a very small arithmetic subset safely.

    Supports +, -, *, /, ^ and parentheses with floats. Rejects other characters.
    """
    if not re.fullmatch(r"[0-9eE+\-*/^().\s]+", expression):
        return "Invalid characters in expression."

    # Tokenize numbers, operators, parentheses
    tokens = re.findall(r"\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?|[()+\-*/^]", expression)
    if not tokens:
        return "Empty expression."

    # Shunting-yard to RPN
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    right_assoc = {"^"}
    output: list[str] = []
    ops: list[str] = []

    for tok in tokens:
        if re.match(r"\d", tok):
            output.append(tok)
        elif tok in ALLOWED_OPERATORS:
            while (
                ops
                and ops[-1] in ALLOWED_OPERATORS
                and (
                    (tok not in right_assoc and precedence[ops[-1]] >= precedence[tok])
                    or (tok in right_assoc and precedence[ops[-1]] > precedence[tok])
                )
            ):
                output.append(ops.pop())
            ops.append(tok)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                return "Mismatched parentheses."
            ops.pop()
        else:
            return "Unsupported token."

    while ops:
        op = ops.pop()
        if op in ("(", ")"):
            return "Mismatched parentheses."
        output.append(op)

    # Evaluate RPN
    stack: list[float] = []
    try:
        for tok in output:
            if tok in ALLOWED_OPERATORS:
                if len(stack) < 2:
                    return "Invalid expression."
                b = stack.pop()
                a = stack.pop()
                # Protect against pow with huge exponents
                if tok == "^" and abs(b) > 1e6:
                    return "Exponent too large."
                stack.append(float(ALLOWED_OPERATORS[tok](a, b)))
            else:
                stack.append(float(tok))
        if len(stack) != 1:
            return "Invalid expression."
        result = stack[0]
        if math.isinf(result) or math.isnan(result):
            return "Computation resulted in non-finite value."
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


# Nodes
def router_node(state: ChatState) -> ChatState:
    text = (state.get("message") or "").strip()
    intent: Literal["search", "calc", "chat"]
    if re.search(r"\b(search|find|lookup|web)\b", text, flags=re.I):
        intent = "search"
    elif re.search(r"[0-9]" , text) and re.search(r"[+\-*/^]", text):
        intent = "calc"
    else:
        intent = "chat"
    return {"step": intent.upper(), "intent": intent}


def search_node(state: ChatState) -> ChatState:
    query = state.get("message", "").strip()
    try:
        result = web_search(query)
        return {"tool_result": result, "step": "RESPOND"}
    except Exception as exc:
        return {"error": f"Search failed: {exc}", "step": "RESPOND"}


def calc_node(state: ChatState) -> ChatState:
    expr = state.get("message", "").strip()
    result = safe_calculate(expr)
    return {"tool_result": result, "step": "RESPOND"}


def chat_node(state: ChatState) -> ChatState:
    # Very naive fallback answer
    return {
        "tool_result": "I'm a tool-enabled bot. Ask me to search the web or do math.",
        "step": "RESPOND",
    }


def respond_node(state: ChatState) -> ChatState:
    return {"step": "ROUTER"}


# Build graph
graph = StateGraph(ChatState)
graph.add_node("ROUTER", router_node)
graph.add_node("SEARCH", search_node)
graph.add_node("CALC", calc_node)
graph.add_node("CHAT", chat_node)
graph.add_node("RESPOND", respond_node)

graph.set_entry_point("ROUTER")

# Conditional edges from router
def route_selector(state: ChatState) -> str:
    intent = state.get("intent", "chat")
    return intent.upper()


graph.add_conditional_edges("ROUTER", route_selector, {"SEARCH": "SEARCH", "CALC": "CALC", "CHAT": "CHAT"})

# After any tool/chat, go to RESPOND, then END this turn
graph.add_edge("SEARCH", "RESPOND")
graph.add_edge("CALC", "RESPOND")
graph.add_edge("CHAT", "RESPOND")
graph.add_edge("RESPOND", END)

app = graph.compile()


def main() -> None:
    # One-shot CLI mode: pass a query as arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        state: ChatState = {"step": "ROUTER", "message": query}
        state = app.invoke(state)
        output = state.get("tool_result") or state.get("error") or "(no output)"
        print(output)
        return
    print("Tool-enabled LangGraph Chatbot")
    print("=" * 40)
    print("Hints: 'search what is langgraph', '2 + 2 * 3', or general chat.")
    state: ChatState = {"step": "ROUTER", "message": ""}
    try:
        while True:
            user_in = input("You: ")
            state["message"] = user_in
            # Run one turn: router -> (tool/chat) -> respond -> router
            state = app.invoke(state)
            # In this simple design, the last node stores output in tool_result
            output = state.get("tool_result") or state.get("error") or "(no output)"
            print("Bot:", output)
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")


if __name__ == "__main__":
    main()


