from __future__ import annotations

from typing import Callable, Dict, Any, Optional, List, TypedDict
import os
import json
import re
import math
import operator
import datetime as dt
import urllib.parse
import urllib.request
import html as html_lib

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore


# ---------- Utilities: Web search (DuckDuckGo HTML) ----------
def web_search(query: str, num_results: int = 5) -> str:
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

    pages: List[str] = []
    try:
        pages.append(fetch_html("https://html.duckduckgo.com"))
    except Exception:
        pass
    try:
        pages.append(fetch_html("https://duckduckgo.com"))
    except Exception:
        pass

    results: List[str] = []
    for html in pages:
        if results:
            break
        patterns = [
            r"<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
            r"<h2[^>]*class=\"result__title\"[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
            r"<a[^>]+class=\"result__url[^\"]*\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, html, flags=re.I | re.S):
                href = m.group(1)
                title_raw = m.group(2)
                title = html_lib.unescape(re.sub("<.*?>", "", title_raw)).strip()
                if not title:
                    continue
                if href.startswith("/l/?"):
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

    return "\n".join(results) if results else "No results found."


# ---------- Utilities: Calculator ----------
ALLOWED_OPERATORS: Dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}


def safe_calculate(expression: str) -> str:
    if not re.fullmatch(r"[0-9eE+\-*/^().\s]+", expression):
        return "Invalid characters in expression."
    tokens = re.findall(r"\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?|[()+\-*/^]", expression)
    if not tokens:
        return "Empty expression."
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
    right_assoc = {"^"}
    output: List[str] = []
    ops: List[str] = []
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
    stack: List[float] = []
    try:
        for tok in output:
            if tok in ALLOWED_OPERATORS:
                if len(stack) < 2:
                    return "Invalid expression."
                b = stack.pop()
                a = stack.pop()
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


# ---------- Utilities: Datetime ----------
def current_datetime(_: str = "") -> str:
    now = dt.datetime.now()
    return now.isoformat(sep=" ", timespec="seconds")


# ---------- Utilities: Simple in-memory docs ----------
SIMPLE_DOCS: Dict[str, str] = {
    "project": "This repo contains multiple chatbot examples (LangGraph, tools, LLM memory).",
    "tools": "Tools include web search (DuckDuckGo), calculator, datetime, and docs lookup.",
}


def lookup_doc(query: str) -> str:
    q = query.lower().strip()
    for key, value in SIMPLE_DOCS.items():
        if key in q:
            return f"{key}: {value}"
    return "No matching docs."


# ---------- Redis Memory (Upstash REST preferred) ----------
class MemoryStore:
    def __init__(self) -> None:
        self.upstash_url = os.environ.get("UPSTASH_REDIS_REST_URL")
        self.upstash_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        self.redis_url = os.environ.get("REDIS_URL")
        self.ttl_seconds = int(os.environ.get("CHAT_TTL_SECONDS", "86400"))
        self.window = int(os.environ.get("CHAT_RECENT_WINDOW", "12"))
        # Optional redis-py fallback
        self.redis_client = None
        if not self.upstash_url and self.redis_url:
            try:
                import redis  # type: ignore

                self.redis_client = redis.from_url(self.redis_url)
            except Exception:
                self.redis_client = None

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}:messages"

    # Upstash REST helpers
    def _upstash_cmd(self, *parts: str) -> Optional[Any]:
        assert self.upstash_url and self.upstash_token
        path = "/".join(urllib.parse.quote(p, safe="") for p in parts)
        url = f"{self.upstash_url}/{path}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.upstash_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(data)
        except Exception:
            return data

    def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        entry = json.dumps({"role": role, "content": content})
        if self.upstash_url and self.upstash_token:
            self._upstash_cmd("LPUSH", key, entry)
            self._upstash_cmd("LTRIM", key, "0", str(self.window - 1))
            self._upstash_cmd("EXPIRE", key, str(self.ttl_seconds))
            return
        if self.redis_client is not None:
            self.redis_client.lpush(key, entry)
            self.redis_client.ltrim(key, 0, self.window - 1)
            self.redis_client.expire(key, self.ttl_seconds)
            return

    def recent(self, session_id: str) -> List[Dict[str, str]]:
        key = self._key(session_id)
        raw: List[str] = []
        if self.upstash_url and self.upstash_token:
            res = self._upstash_cmd("LRANGE", key, "0", str(self.window - 1))
            if isinstance(res, dict) and "result" in res and isinstance(res["result"], list):
                raw = list(reversed(res["result"]))
        elif self.redis_client is not None:
            vals = self.redis_client.lrange(key, 0, self.window - 1)
            raw = [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in reversed(vals)]
        results: List[Dict[str, str]] = []
        for item in raw:
            try:
                obj = json.loads(item)
                if isinstance(obj, dict) and "role" in obj and "content" in obj:
                    results.append({"role": obj["role"], "content": obj["content"]})
            except Exception:
                continue
        return results


# ---------- Tool registry ----------
class Tool(TypedDict):
    name: str
    description: str
    func: Callable[[str], str]


TOOLS: List[Tool] = [
    {
        "name": "web_search",
        "description": "Search the web for up-to-date information. Input: search query string.",
        "func": lambda q: web_search(q, num_results=5),
    },
    {
        "name": "calculator",
        "description": "Safely evaluate arithmetic expressions (+, -, *, /, ^, parentheses). Input: expression string.",
        "func": safe_calculate,
    },
    {
        "name": "datetime",
        "description": "Get the current local date and time in ISO format. Input: ignored.",
        "func": current_datetime,
    },
    {
        "name": "docs_lookup",
        "description": "Lookup simple project docs by keyword (e.g., 'project', 'tools'). Input: keyword string.",
        "func": lookup_doc,
    },
]

NAME_TO_TOOL: Dict[str, Tool] = {t["name"]: t for t in TOOLS}


# ---------- ReAct-like Agent ----------
def llm_available() -> bool:
    return os.environ.get("GOOGLE_API_KEY") is not None and genai is not None


def format_tools_description() -> str:
    lines = []
    for t in TOOLS:
        lines.append(f"- {t['name']}: {t['description']}")
    return "\n".join(lines)


def agent_decide_and_act(message: str, memory: MemoryStore, session_id: str) -> str:
    recent = memory.recent(session_id)
    memory.append(session_id, "user", message)

    if llm_available():
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash"))
        system = (
            "You are a ReAct agent. Decide if a tool is needed. "
            "Tools available:\n" + format_tools_description() + "\n" \
            "Return a JSON object: {\"action\": one of [" \
            + ", ".join(f'\"{t["name"]}\"' for t in TOOLS) + ", \"final\"], " \
            "\"input\": string with the tool input or final answer}."
        )
        msgs = [system]
        for m in recent:
            if m["role"] == "user":
                msgs.append(f"User: {m['content']}")
            else:
                msgs.append(f"Assistant: {m['content']}")
        msgs.append(f"User: {message}")
        prompt = "\n".join(msgs)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=300,
        ))
        content = response.text or "{}"
        # Clean up markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        try:
            plan = json.loads(content)
        except Exception:
            plan = {"action": "final", "input": content}

        action = str(plan.get("action", "final"))
        action_input = str(plan.get("input", ""))
        if action in NAME_TO_TOOL:
            tool = NAME_TO_TOOL[action]
            tool_out = tool["func"](action_input)
            # Ask model to formulate final answer including tool result
            follow_prompt = prompt + f"\nAssistant: TOOL[{action}] -> {tool_out}\nUser: Please provide a final answer based on the tool result."
            response2 = model.generate_content(follow_prompt, generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
            ))
            final = response2.text or tool_out
            memory.append(session_id, "assistant", final)
            return final
        else:
            final = action_input if action == "final" else str(plan)
            memory.append(session_id, "assistant", final)
            return final

    # Fallback deterministic router without LLM
    text = message.strip()
    if re.search(r"\b(search|find|lookup|web)\b", text, flags=re.I):
        out = NAME_TO_TOOL["web_search"]["func"](text)
    elif re.search(r"[0-9]", text) and re.search(r"[+\-*/^]", text):
        out = NAME_TO_TOOL["calculator"]["func"](text)
    elif re.search(r"\b(time|date|day|clock)\b", text, flags=re.I):
        out = NAME_TO_TOOL["datetime"]["func"](text)
    else:
        out = NAME_TO_TOOL["docs_lookup"]["func"](text)
    memory.append(session_id, "assistant", out)
    return out


def main() -> None:
    print("ReAct Agent Chatbot (Tools + Redis/Upstash Memory)")
    print("=" * 40)
    print("Env: set GOOGLE_API_KEY for Gemini LLM. Type 'exit' to quit.")
    session_id = os.environ.get("CHAT_SESSION_ID", "default")
    memory = MemoryStore()
    try:
        while True:
            user_in = input("You: ")
            if user_in.strip().lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye!")
                break
            reply = agent_decide_and_act(user_in, memory, session_id)
            print("Bot:", reply)
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")


if __name__ == "__main__":
    main()


