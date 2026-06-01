import os, json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from groq import Groq

load_dotenv()

@tool
def check_transition(from_camelot: str, to_camelot: str) -> dict:
    """Check harmonic smoothness between two Camelot positions."""
    return {"score": 0.8, "verdict": "smooth"}

PROMPT = "Reason step by step about whether 8B to 9B is a smooth DJ transition, then call the tool."

# ---- A) ChatGroq: dump the WHOLE message, don't guess keys ----
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3,
               reasoning_format="parsed").bind_tools([check_transition])
msg = llm.invoke([HumanMessage(PROMPT)])
print("=== ChatGroq ===")
print("additional_kwargs:", json.dumps(msg.additional_kwargs, indent=2, default=str))
print("response_metadata:", json.dumps(msg.response_metadata, indent=2, default=str))
print("tool_calls:", msg.tool_calls)

# ---- B) Raw Groq SDK: identical call, inspect message.reasoning ----
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
raw = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    messages=[{"role": "user", "content": PROMPT}],
    tools=[{"type": "function", "function": {
        "name": "check_transition",
        "description": "Check harmonic smoothness between two Camelot positions.",
        "parameters": {"type": "object", "properties": {
            "from_camelot": {"type": "string"}, "to_camelot": {"type": "string"}},
            "required": ["from_camelot", "to_camelot"]}}}],
    tool_choice="auto",
)
m = raw.choices[0].message
print("\n=== Raw Groq SDK ===")
print("content :", repr(m.content))
print("reasoning:", repr(getattr(m, "reasoning", "<no attr>")))
print("tool_calls:", m.tool_calls)