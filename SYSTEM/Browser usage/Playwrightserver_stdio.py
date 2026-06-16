import json
import subprocess
import threading
import time
import sys
import requests
from openai import OpenAI

# ── NIM client ────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)
NIM_MODEL = "meta/llama-3.1-8b-instruct"

class MCPClient:

    def __init__(self):
        self._proc    = None
        self._lock    = threading.Lock()
        self._req_id  = 0
        self._tools   = []          # cached list of MCP tool defs

    def start(self):
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities":    {},
            "clientInfo":      {"name": "nim-browser-agent", "version": "1.0"},
        })

        # Notification (no id, no response expected)
        notif   = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        headers = {
            "Content-Type": "application/json",
            "Accept":       "application/json, text/event-stream",  # ← add here too
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        requests.post(self.base_url, json=notif, headers=headers, timeout=10)
        print(f"[MCP] Connected  session={self._session_id}")

    def stop(self):
        if self._proc:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait()
            print("[MCP] Server stopped.")

    def _drain_stderr(self):
        for line in self._proc.stderr:
            print(f"[MCP stderr] {line.rstrip()}", file=sys.stderr)

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send(self, payload: dict) -> dict:
        """Send one JSON-RPC request, return the response dict."""
        with self._lock:
            line = json.dumps(payload) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            raw = self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("MCP server closed stdout unexpectedly.")
            return json.loads(raw)

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id":      self._next_id(),
            "method":  method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept":       "application/json, text/event-stream",  # ← this was missing
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]

        # Server may respond with SSE stream instead of plain JSON
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(resp.text)

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error [{method}]: {data['error']}")
        return data.get("result", {})

    def _parse_sse(self, raw: str) -> dict:
        """Parse SSE response — extract the JSON-RPC result from data: lines."""
        for line in raw.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                    if "error" in data:
                        raise RuntimeError(f"MCP SSE error: {data['error']}")
                    return data.get("result", {})
                except json.JSONDecodeError:
                    continue
        return {}

    def _initialize(self):
        """MCP initialize + initialized handshake."""
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities":    {},
            "clientInfo":      {"name": "nim-browser-agent", "version": "1.0"},
        })
        # Send the required initialized notification (no id)
        notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self._proc.stdin.write(notif)
        self._proc.stdin.flush()

    def list_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling schema."""
        if self._tools:
            return self._tools

        result = self._rpc("tools/list")
        raw_tools = result.get("tools", [])

        # Convert MCP schema → OpenAI tool schema
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in raw_tools
        ]
        return self._tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call a Playwright MCP tool, return the text result."""
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # Result is a list of content blocks; grab the text ones
        content = result.get("content", [])
        parts   = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(result)


# ── Agent loop ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a browser automation agent powered by Playwright MCP tools.
You are given a GOAL and control a real Chrome browser through tool calls.

Rules:
- The browser may already be logged in — NEVER type passwords or try to log in.
- If you see a login/CAPTCHA page, call browser_wait_for_timeout and then retry.
- Call browser_navigate to go to a URL.
- Call browser_snapshot to inspect the current page (returns accessibility tree).
- Use the snapshot to find the best element reference, then act on it.
- When the goal is fully achieved, stop calling tools and say "GOAL ACHIEVED: <summary>".
- If stuck after 3 retries on the same step, say "STUCK: <reason>" and stop.
- Never loop — vary your approach if a tool call fails.
"""

def run_agent(goal: str, start_url: str, max_steps: int = 25):
    mcp = MCPClient()
    mcp.start(headless=False)          # set True for headless

    tools    = mcp.list_tools()
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": (
            f"GOAL: {goal}\n"
            f"Start by navigating to: {start_url or 'https://www.google.com'}"
        )},
    ]

    print(f"\nGoal   : {goal}")
    print(f"URL    : {start_url or 'https://www.google.com'}")
    print(f"Tools  : {[t['function']['name'] for t in tools]}")
    print("=" * 60)

    try:
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} / {max_steps} ---")

            response = client.chat.completions.create(
                model      = NIM_MODEL,
                messages   = messages,
                tools      = tools,
                tool_choice= "auto",
                max_tokens = 512,
                temperature= 0,
            )
            msg = response.choices[0].message

            # ── text reply (possibly terminal) ───────────────────────────────
            if msg.content:
                print(f"[LLM] {msg.content}")
                if "GOAL ACHIEVED" in msg.content or "STUCK" in msg.content:
                    break

            # ── no tool call → LLM is done ───────────────────────────────────
            if not msg.tool_calls:
                print("[LLM] No tool call — agent finished.")
                break

            # ── execute each tool call ────────────────────────────────────────
            messages.append(msg)           # add assistant turn (with tool_calls)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                print(f"  → Tool : {fn_name}")
                print(f"    Args : {json.dumps(fn_args, ensure_ascii=False)[:200]}")

                tool_result = mcp.call_tool(fn_name, fn_args)

                # Truncate long snapshots so we don't overflow context
                if len(tool_result) > 3000:
                    tool_result = tool_result[:3000] + "\n...[truncated]"

                print(f"    Result (first 300 chars): {tool_result[:300]}")

                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "content":     tool_result,
                })

        else:
            print("\nMax steps reached.")

    finally:
        input("\nPress Enter to close the browser...")
        mcp.stop()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    goal      = input("Enter your goal : ").strip()
    start_url = input("Starting URL    : ").strip()
    run_agent(goal, start_url)