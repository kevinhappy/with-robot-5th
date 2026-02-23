import os
import re
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_fireworks import ChatFireworks
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Dict

import google.generativeai as genai

# Load environment variables
load_dotenv()
# api_key = os.getenv("FIREWORKS_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ 경고: FIREWORKS_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
else:
    print(f"✅ API Key 로드 성공 (앞 4자리): {api_key[:4]}****")

ROBOT_URL = "http://127.0.0.1:8800"

# Initialize LLM
# llm = ChatFireworks(
#     # model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
#     # model="accounts/fireworks/models/qwen2p5-72b-instruct",
#     model="accounts/kevinhappy2-ac59ff/deployments/zm2zrk70",
#     fireworks_api_key=api_key, # 명시적으로 전달
#     max_tokens=1000
# )

# genai.configure(api_key=api_key)
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",  # 'models/'를 앞에 붙여줍니다.
    google_api_key=api_key,   # Gemini API Key 전달
    max_output_tokens=1000,
    temperature=0.7
)

# Load code repository once at module initialization
CODE_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "robot", "code_knowledge.md")
def load_code_knowledge():
    """Load robot control API documentation."""
    try:
        with open(CODE_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Error: code_knowledge.md not found"
# Load once at module level
CODE_KNOWLEDGE = load_code_knowledge()


class State(TypedDict):
    """AI Agent conversation state."""
    generated_code: str # Generated code
    exec_result: dict # Execution result
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]  # Conversation history


def plan_node(state: State) -> State:
    """Create Robot Control code."""
    try:
        payload = {"action": {
                        "type": "run_code",
                        "payload": {
                            "code": """objects = get_object_positions()
RESULT["objects"] = objects"""
                        }
                    }}
        response = requests.post(f"{ROBOT_URL}/send_action", json=payload)
        # print(response)
        objects = response.json()["result"]["objects"]
        # print(objects)
        objects_str = json.dumps(objects, indent=2)
        # print(objects_str)

        system_prompt = f"""You are a helpful robot assistant.

Generate Python code based on the user's command using the API below.

## Scene Context (Available Objects)
{objects_str}

## Rules
- Return ONLY executable Python code in a single generic block used markdown.
- NO imports allowed (`time`, `math`, `list` are pre-loaded).
- Use `RESULT` dict for return values.
- Be concise and natural.
- When referring to objects, use the exact names from Scene Context.

## API Reference
{CODE_KNOWLEDGE}
```"""

        generated_code = llm.invoke([
                                        SystemMessage(content=system_prompt)
                                    ] + state["messages"])
        # Extract code using regex
        print(generated_code)

        # The 'content' from the LLM can be a string or a list of parts.
        content_str = ""
        if isinstance(generated_code.content, list) and len(generated_code.content) > 0:
            content_str = generated_code.content[0].get("text", "")
        elif isinstance(generated_code.content, str):
            content_str = generated_code.content

        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", content_str, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        
        return {"generated_code": generated_code}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"messages": [traceback.format_exc()], "generated_code": None}


def exec_node(state: State) -> State:
    """Execute Robot Control code."""
    try:
        generated_code = state["generated_code"]
        
        payload = {"action": {
                "type": "run_code",
                "payload": {
                    "code": generated_code
                }
            }}
        exec_result = requests.post(f"{ROBOT_URL}/send_action", json=payload).json()
        
        return {"exec_result": exec_result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"messages": [traceback.format_exc()], "exec_result": None}


def create_graph(checkpointer=None):
    """Create chat graph with optional checkpointer for memory."""
    graph_builder = (
        StateGraph(State)
        .add_node("plan", plan_node)
        .add_node("exec", exec_node)
        .add_edge(START, "plan")
        .add_edge("plan", "exec")
        .add_edge("exec", END)
    )
    
    if checkpointer:
        return graph_builder.compile(checkpointer=checkpointer)
    else:
        return graph_builder.compile()


__all__ = ["graph", "create_graph"]
