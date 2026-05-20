"""
Tests for the Groq LLM connection.

Requires GROQ_API_KEY in .env and a live network connection.

Test 1 — basic completion: verify the API key is valid and the model responds.
Test 2 — JSON response: verify the model can follow a structured output prompt.
Test 3 — tool calling: verify the model can declare a tool call in the response.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def run_tests():
    passed = 0
    failed = 0

    def check(label, condition, got=None, expected=None):
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {label}")
            passed += 1
        else:
            print(f"  ✗  {label}  (got {got!r}, expected {expected!r})")
            failed += 1

    api_key = os.getenv("GROQ_API_KEY")

    print("\n" + "=" * 55)
    print(f"Groq LLM — Connection Tests  (model: {MODEL})")
    print("=" * 55)
    print(f"  GROQ_API_KEY: {'SET' if api_key else 'MISSING'}")

    if not api_key:
        print("\n  Skipping all tests — GROQ_API_KEY not set in .env\n")
        return False

    client = Groq(api_key=api_key)

    # ── Test 1: basic completion ──────────────────────────────────────────────
    print("\n[1] Basic completion — model responds to a simple prompt")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content or ""
        check("response is non-empty",       bool(text.strip()))
        check("finish_reason is 'stop'",     resp.choices[0].finish_reason == "stop",
              got=resp.choices[0].finish_reason, expected="stop")
        check("model field matches",         MODEL in resp.model,
              got=resp.model, expected=MODEL)
        print(f"      reply: {text.strip()!r}")
    except Exception as exc:
        print(f"  ✗  [1] exception: {exc}")
        failed += 3

    # ── Test 2: JSON output ───────────────────────────────────────────────────
    print("\n[2] Structured JSON response — DJ track selection")
    prompt = """Current listener state:
- Energy: 0.75 (building — wants more energy)
- Valence: 0.60 (positive mood)
- Session arc: BUILD phase — approaching peak

Current track: Midnight City — M83 (synthpop, 105 BPM, Ab major)

Candidate tracks:
1. Time — Pink Floyd  (progressive rock, 123 BPM, D major)
2. Pumped Up Kicks — Foster the People  (indie pop, 111 BPM, F major)
3. Retrograde — James Blake  (electronic, 66 BPM, Db major)
4. Mr. Brightside — The Killers  (indie rock, 148 BPM, G major)

Select the best next track. Respond with JSON only — no markdown, no code blocks:
{"selected": "<track name>", "artist": "<artist>", "reason": "<1-2 sentences>"}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are the reasoning core of an Agentic DJ. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        raw = resp.choices[0].message.content or ""
        print(f"      raw: {raw.strip()!r}")

        try:
            result = json.loads(raw.strip())
            check("response parses as JSON",   True)
            check("'selected' key present",    "selected" in result,  got=list(result.keys()))
            check("'artist' key present",      "artist"   in result,  got=list(result.keys()))
            check("'reason' key present",      "reason"   in result,  got=list(result.keys()))
            check("selected is a string",      isinstance(result.get("selected"), str))
            print(f"      selected: {result.get('selected')} — {result.get('artist')}")
            print(f"      reason:   {result.get('reason')}")
        except json.JSONDecodeError as e:
            print(f"  ✗  JSON parse failed: {e}")
            failed += 5

    except Exception as exc:
        print(f"  ✗  [2] exception: {exc}")
        failed += 5

    # ── Test 3: tool calling ──────────────────────────────────────────────────
    print("\n[3] Tool calling — model declares a function call")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "select_track",
                "description": "Select a track to queue next.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track_name": {"type": "string", "description": "Track title"},
                        "artist":     {"type": "string", "description": "Artist name"},
                        "reason":     {"type": "string", "description": "Why this track"},
                    },
                    "required": ["track_name", "artist", "reason"],
                },
            },
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a DJ assistant. Use the select_track tool to pick the next song.",
                },
                {
                    "role": "user",
                    "content": "The listener wants high-energy electronic music. Pick a track.",
                },
            ],
            tools=tools,
            tool_choice="required",
            max_tokens=150,
        )
        msg = resp.choices[0].message
        tc  = (msg.tool_calls or [None])[0]

        check("tool_calls present",            bool(msg.tool_calls))
        check("function name is 'select_track'",
              tc is not None and tc.function.name == "select_track",
              got=tc.function.name if tc else None, expected="select_track")

        if tc:
            args = json.loads(tc.function.arguments)
            check("'track_name' in args",      "track_name" in args, got=list(args.keys()))
            check("'artist' in args",          "artist"     in args, got=list(args.keys()))
            check("'reason' in args",          "reason"     in args, got=list(args.keys()))
            print(f"      track:  {args.get('track_name')} — {args.get('artist')}")
            print(f"      reason: {args.get('reason')}")
        else:
            failed += 3

    except Exception as exc:
        print(f"  ✗  [3] exception: {exc}")
        failed += 5

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'=' * 55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
