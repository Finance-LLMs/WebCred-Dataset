import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch


import os

os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here" # Replace with your actual Tavily API key




# Load dataset

def load_dataset(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

dataset = load_dataset("website_dataset.json")

# Sanity check
labels = {item["label"] for item in dataset}
print("Labels found:", labels)



# Prompt Construction

def build_prompt(entry: Dict) -> str:
    return f"""
You are given a topic and source URL.

Topic: {entry['topic']}
Source URL: {entry['url']}

Task:
1) Use web_search tool to gather up-to-date, relevant snippets.
2) Explain the key claims related to this topic.
3) If the information appears unreliable, incomplete, or misleading, explain why.
4) If you are uncertain, state the uncertainty explicitly.

Respond concisely.
""".strip()


# Model + Tools (LangChain style, per Tool Usage -- Web Search)

NVIDIA_API_KEY = "your_nvidia_api_key_here"  # Replace with your actual API key or set as environment variable
if not NVIDIA_API_KEY:
    raise RuntimeError("Please set NVIDIA_API_KEY in your environment to call the NIM endpoint.")

MODEL_NAME = "meta/llama-3.1-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Chat model
llm = ChatNVIDIA(model=MODEL_NAME, api_key=NVIDIA_API_KEY, base_url=BASE_URL, temperature=0.2)

# Tavily web search tool (requires TAVILY_API_KEY)
if "TAVILY_API_KEY" not in os.environ:
    raise RuntimeError("Please set TAVILY_API_KEY for Tavily web search tool calls.")

tavily_search = TavilySearch(max_results=3)
tools_by_name = {tavily_search.name: tavily_search}

# Bind tools to the model so it can decide to call web search
llm_with_tools = llm.bind_tools([tavily_search])


def query_model_with_web(entry: Dict, prompt: str) -> str:
    """Use tool-calling to run web search then answer."""

    system_msg = SystemMessage(
        content=(
            "You are a careful assistant. Use the web_search tool when you need current or source information. "
            "Cite the URL snippets you use and state uncertainty when unsure."
        )
    )
    user_msg = HumanMessage(content=prompt)

    ai_msg = llm_with_tools.invoke([system_msg, user_msg])

    # If the model decides no tool is needed
    if not getattr(ai_msg, "tool_calls", None):
        return ai_msg.content

    tool_messages = []
    for call in ai_msg.tool_calls:
        tool = tools_by_name.get(call["name"])
        if not tool:
            continue
        try:
            result = tool.invoke(call["args"])
        except Exception as exc:  # Keep evaluation running even if a call fails
            result = {"error": str(exc)}
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    final_system_msg = SystemMessage(
    content=(
        "You have gathered web search results. "
        "Now write a concise final answer to the user, citing sources where relevant. "
        "Do not call any more tools."
    )
)

    final_msg = llm_with_tools.invoke(
    [system_msg, user_msg, ai_msg, *tool_messages, final_system_msg]
)

    return final_msg.content



# Run Evaluation

def run_evaluation(dataset, sample_size=100):
    results = []

    sample_size = min(sample_size, len(dataset))
    sampled_data = random.sample(dataset, sample_size)

    for entry in sampled_data:
        prompt = build_prompt(entry)
        output = query_model_with_web(entry, prompt)

        results.append({
            "id": str(uuid.uuid4()),
            "topic": entry["topic"],
            "url": entry["url"],
            "gold_label": entry["label"],   # not shown to model
            "prompt": prompt,
            "model_output": output,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    return results

results = run_evaluation(dataset, sample_size=100)


# Save Results

def save_results(results, path="model_behavior_outputs.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

save_results(results)
