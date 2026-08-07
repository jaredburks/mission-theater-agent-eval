"""
LLM agent adapters. These are the ONLY part of the repo that needs credentials.
The env, baselines, harness, metrics, and report all run without them.

Each adapter implements act(obs, last_events) -> action dict by:
  1. rendering the observation into a prompt,
  2. calling the model with TOOL_SCHEMA,
  3. parsing the tool_use block back into an action.

All three adapters (Anthropic, Ollama, OpenAI) implement the same interface;
adding a provider means adding one class with one method.
"""
from __future__ import annotations

import os

from .interface import (TOOL_SCHEMA, SYSTEM_PROMPT, render_observation,
                        parse_action, to_openai_tools)


def _extract_tool_calls(resp):
    """Robust across ollama-python versions: handle object- and dict-shaped responses."""
    msg = getattr(resp, "message", None) or (resp.get("message") if isinstance(resp, dict) else None)
    if msg is None:
        return [], ""
    tool_calls = getattr(msg, "tool_calls", None)
    content = getattr(msg, "content", None)
    if tool_calls is None and isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
        content = msg.get("content")
    return tool_calls or [], content or ""


class AnthropicAgent:
    """Cloud model via the Anthropic API.

    Multi-turn: keeps the per-episode conversation (assistant tool_use ->
    tool_result -> next observation), mirroring OllamaAgent. Stateless-per-turn
    was tried first and scan-looped/dithered — same cliff the local models hit
    without history. Episodes are <=45 turns, so the full history fits context
    easily; incremental prompt caching keeps the token bill near-flat.
    """
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        # Imported lazily so the repo imports cleanly without the SDK installed.
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens
        self._messages: list[dict] = []
        self._pending_tool_use_id: str | None = None

    def act(self, obs, last_events=None):
        system = SYSTEM_PROMPT.format(objective=obs["objective"])
        user_text = render_observation(obs) + \
            "\n\nCall exactly one tool for your next action."

        content: list[dict] = []
        if self._pending_tool_use_id is not None:
            # Close out the previous tool call with its environment result.
            result_text = "\n".join(last_events) if last_events else "ok"
            content.append({"type": "tool_result",
                            "tool_use_id": self._pending_tool_use_id,
                            "content": result_text})
            self._pending_tool_use_id = None
        content.append({"type": "text", "text": user_text,
                        "cache_control": {"type": "ephemeral"}})
        self._strip_stale_cache_marks()
        self._messages.append({"role": "user", "content": content})

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=TOOL_SCHEMA,
            tool_choice={"type": "any"},
            messages=self._messages,
        )
        # Persist the assistant turn so the next call sees the full history.
        kept = []
        for b in resp.content:
            btype = getattr(b, "type", None)
            if btype == "tool_use":
                kept.append({"type": "tool_use", "id": b.id,
                             "name": b.name, "input": b.input})
            elif btype == "text" and b.text:
                kept.append({"type": "text", "text": b.text})
        if not kept:  # keep history well-formed even on an empty reply
            kept = [{"type": "text", "text": "(no action)"}]
        self._messages.append({"role": "assistant", "content": kept})
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                self._pending_tool_use_id = block.id
                return parse_action({"name": block.name, "input": block.input})
        return {"tool": "noop"}

    def _strip_stale_cache_marks(self):
        """Keep only the newest user-message cache breakpoint (limit is 4 total;
        system uses one, the incremental conversation prefix uses another)."""
        for msg in self._messages:
            if msg["role"] != "user":
                continue
            for block in msg["content"]:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


class OllamaAgent:
    """Local model via Ollama. Runs against the OpenAI-compatible tool API.
    Qwen3/3.5 and Gemma4 are thinking models — leave think=False for reliable,
    faster tool calls (thinking blocks have been known to corrupt multi-turn tools).

    Keeps a per-episode message history (assistant tool_call -> tool result ->
    next observation) so the model isn't stuck replaying the opening scan every turn.
    """

    def __init__(self, model: str = "qwen3:14b", host: str = "http://localhost:11434",
                 temperature: float = 0.0, label: str | None = None,
                 think: bool = False):
        from ollama import Client  # lazy import; repo loads without the package
        self.client = Client(host=host)
        self.model = model
        self.temperature = temperature
        self.think = think
        self.tools = to_openai_tools()
        self.name = label or model.replace(":", "-")
        self._messages: list[dict] | None = None
        self._pending_call: dict | None = None

    def act(self, obs, last_events=None):
        system = SYSTEM_PROMPT.format(objective=obs["objective"])
        if self._messages is None:
            self._messages = [{"role": "system", "content": system}]

        # Close out the previous tool call with its environment result.
        if self._pending_call is not None:
            self._messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [self._pending_call],
            })
            result_text = "\n".join(last_events) if last_events else "ok"
            fn = self._pending_call.get("function") or {}
            self._messages.append({
                "role": "tool",
                "tool_name": fn.get("name", "tool"),
                "content": result_text,
            })
            self._pending_call = None

        user = render_observation(obs)
        user += ("\n\nCall exactly one tool for your next action. "
                 "Do not call more than one. If the map still has '?' cells, "
                 "MOVE a unit toward them instead of rescanning a known area.")
        self._messages.append({"role": "user", "content": user})
        self._trim_history()

        kwargs = dict(
            model=self.model,
            messages=self._messages,
            tools=self.tools,
            options={"temperature": self.temperature, "num_ctx": 8192},
        )
        try:
            resp = self.client.chat(**kwargs, think=self.think)
        except TypeError:
            resp = self.client.chat(**kwargs)

        tool_calls, content = _extract_tool_calls(resp)
        if tool_calls:
            call = tool_calls[0]
            fn = getattr(call, "function", None) or call.get("function", {})
            name = getattr(fn, "name", None) or fn.get("name")
            args = getattr(fn, "arguments", None) or fn.get("arguments", {})
            args = dict(args) if args else {}
            # Persist in OpenAI-shaped dict so the next turn can feed a tool result.
            self._pending_call = {
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            return parse_action({"name": name, "input": args})
        return parse_action(content) if content else {"tool": "noop"}

    def _trim_history(self, max_messages: int = 24):
        """Keep system + a recent window so long episodes don't blow the context."""
        if self._messages is None or len(self._messages) <= max_messages:
            return
        system, rest = self._messages[0], self._messages[1:]
        self._messages = [system] + rest[-(max_messages - 1):]


class OpenAIAgent:
    """Cloud model via the OpenAI API. Same converter, same parse path."""

    def __init__(self, model: str = "gpt-4.1", label: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.tools = to_openai_tools()
        self.name = label or model

    def act(self, obs, last_events=None):
        system = SYSTEM_PROMPT.format(objective=obs["objective"])
        user = render_observation(obs, last_events) + \
            "\n\nCall exactly one tool for your next action."
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            tools=self.tools, tool_choice="required",
        )
        calls = resp.choices[0].message.tool_calls
        if calls:
            import json as _json
            c = calls[0]
            return parse_action({"name": c.function.name,
                                 "input": _json.loads(c.function.arguments)})
        return {"tool": "noop"}
