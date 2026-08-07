"""
One-shot preflight: confirm each candidate advertises `tools` via `ollama show`,
then fire a single add(2,3) tool-call smoke test through the same OllamaAgent path
the benchmark uses. Run before a full suite so you don't burn 30 min on a dud.

    python verify_models.py
"""
from __future__ import annotations

import json
import subprocess
import sys

from agent.interface import to_openai_tools

# Cross-family spread (+ optional small/new and big/old Qwen top-end).
CANDIDATES = [
    ("llama3.1:8b", "llama3.1-8b"),
    ("qwen3:14b", "qwen3-14b"),
    ("gemma4:latest", "gemma4"),
    ("qwen3.5:9b", "qwen3.5-9b"),
    ("qwen3:30b", "qwen3-30b"),
]

ADD_TOOL = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}]


def ollama_show(model: str) -> str:
    r = subprocess.run(
        ["ollama", "show", model],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"ollama show {model} failed")
    return r.stdout


def has_tools_capability(show_text: str) -> bool:
    # Capabilities block lists one capability per line, indented.
    in_caps = False
    for line in show_text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("capabilities"):
            in_caps = True
            continue
        if in_caps:
            if not line.startswith(" ") and stripped:
                break
            if stripped == "tools":
                return True
    return False


def smoke_tool_call(model: str, think: bool = False) -> dict:
    from ollama import Client
    client = Client()
    kwargs = dict(
        model=model,
        messages=[{
            "role": "user",
            "content": "Use the add tool to add 2 and 3. Call the tool; do not answer yourself.",
        }],
        tools=ADD_TOOL,
        options={"temperature": 0.0, "num_ctx": 2048},
    )
    try:
        resp = client.chat(**kwargs, think=think)
    except TypeError:
        resp = client.chat(**kwargs)

    msg = getattr(resp, "message", None) or resp.get("message")
    tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
    if tool_calls is None and isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    content = getattr(msg, "content", None) if msg is not None else None
    if content is None and isinstance(msg, dict):
        content = msg.get("content")

    parsed = []
    for call in tool_calls or []:
        fn = getattr(call, "function", None) or call.get("function", {})
        name = getattr(fn, "name", None) or fn.get("name")
        args = getattr(fn, "arguments", None) or fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parsed.append({"name": name, "arguments": dict(args) if args else {}})
    return {"tool_calls": parsed, "content": content or ""}


def main() -> int:
    print(f"Theater tools schema has {len(to_openai_tools())} actions "
          f"({', '.join(t['function']['name'] for t in to_openai_tools())})")
    print()

    rows = []
    ok_all = True
    for model, label in CANDIDATES:
        print(f"--- {label} ({model}) ---")
        try:
            show = ollama_show(model)
        except RuntimeError as e:
            print(f"  show: FAIL ({e})")
            rows.append((label, model, "missing", "-", "-"))
            ok_all = False
            continue

        tools_cap = has_tools_capability(show)
        print(f"  capabilities.tools: {'yes' if tools_cap else 'NO'}")
        if not tools_cap:
            rows.append((label, model, "no-tools", "-", "-"))
            ok_all = False
            continue

        try:
            result = smoke_tool_call(model, think=False)
        except Exception as e:
            print(f"  smoke: FAIL ({e})")
            rows.append((label, model, "tools", "error", str(e)[:60]))
            ok_all = False
            continue

        calls = result["tool_calls"]
        if calls:
            call = calls[0]
            args = call.get("arguments", {})
            good = call.get("name") == "add" and (
                {str(args.get("a")), str(args.get("b"))} == {"2", "3"}
                or args.get("a") == 2 and args.get("b") == 3
            )
            print(f"  smoke: tool_call {call}")
            rows.append((label, model, "tools", "ok" if good else "weird-args",
                         json.dumps(call)))
            if not good:
                ok_all = False
        else:
            preview = (result["content"] or "")[:120].replace("\n", " ")
            print(f"  smoke: NO tool_call (content={preview!r})")
            rows.append((label, model, "tools", "no-call", preview))
            ok_all = False
        print()

    print("== summary ==")
    print(f"{'label':<14} {'model':<22} {'cap':<10} {'smoke':<12} detail")
    for label, model, cap, smoke, detail in rows:
        print(f"{label:<14} {model:<22} {cap:<10} {smoke:<12} {detail[:50]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
