import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 45)
print("TEST 4 — Google Gemini 3.0 Flash")
print("=" * 45)

prompt = """Current listener state:
- Energy: 0.75 (building — wants more energy)
- Valence: 0.60 (positive mood)
- Session arc: BUILD phase — approaching peak

Current track: Midnight City — M83 (synthpop, dreamy, 105 BPM, Ab major)

Candidate tracks:
1. Time — Pink Floyd  (progressive rock, epic, 123 BPM, D major)
2. Pumped Up Kicks — Foster the People  (indie pop, upbeat, 111 BPM, F major)
3. Retrograde — James Blake  (electronic, melancholic, 66 BPM, Db major)
4. Mr. Brightside — The Killers  (indie rock, energetic, 148 BPM, G major)

Select the best next track given the listener state and session arc.
Respond with JSON only:
{"selected": "<track name>", "artist": "<artist>", "reason": "<1-2 sentences>"}"""

print("\nSending prompt to Gemini 3.0 Flash...")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="""You are the reasoning core of an Agentic DJ system.
You receive the current listener state and candidate tracks,
and select the best next track. Always respond with valid JSON only,
no markdown, no code blocks, no extra text.""",
    ),
)
raw = response.text.strip()

print(f"\n[1] Raw response:\n      {raw}")

# Parse and validate
result = json.loads(raw)
print(f"\n[2] Parsed JSON  ✓")
print(f"      Selected:  {result['selected']} — {result['artist']}")
print(f"      Reason:    {result['reason']}")

print("\n✓ Test 4 passed — Gemini 3.0 Flash working\n")
