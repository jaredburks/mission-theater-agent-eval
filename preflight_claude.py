"""Preflight for the Anthropic API: validates the key, lists available Claude
models, and makes one tiny test call with the configured model.

Run before a benchmark:  python preflight_claude.py
Model override:          set MT_CLAUDE_MODEL=<model-id>  (or put it in .env)
"""
import os
import sys


def _load_dotenv(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except FileNotFoundError:
        pass


def main() -> int:
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FAIL: ANTHROPIC_API_KEY not set (put it in .env)")
        return 1

    from anthropic import Anthropic
    client = Anthropic()

    try:
        models = [m.id for m in client.models.list(limit=100)]
    except Exception as e:
        print(f"FAIL: could not list models: {e}")
        return 1
    print("Available models:")
    for mid in models:
        print(f"  {mid}")

    model = os.environ.get("MT_CLAUDE_MODEL", "claude-sonnet-4-6")
    if model not in models:
        print(f"WARN: configured model '{model}' not in the list above — "
              f"set MT_CLAUDE_MODEL to one that is.")

    try:
        resp = client.messages.create(
            model=model, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the word OK."}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        print(f"Test call to {model}: {text.strip()!r}")
    except Exception as e:
        print(f"FAIL: test call to {model} errored: {e}")
        return 1

    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
