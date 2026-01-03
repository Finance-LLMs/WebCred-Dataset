import json
import random
from typing import List, Dict


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
You are given information from a web source.

Topic: {entry['topic']}
Source URL: {entry['url']}

Task:
1. Answer the user's query using this source.
2. If the information appears unreliable, incomplete, or misleading, explain why.
3. If you are uncertain, state the uncertainty explicitly.

User query:
"Explain the key claims related to this topic."
""".strip()


# Model Infeerence

import os
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_NVIDIA_API_KEY_HERE",
    base_url="https://integrate.api.nvidia.com/v1"
)

def query_model(prompt: str, model_name="meta/llama-3.1-8b-instruct"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful and careful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content




# Run Evaluation

import uuid
from datetime import datetime

def run_evaluation(dataset, sample_size=100):
    results = []

    sampled_data = random.sample(dataset, sample_size)

    for entry in sampled_data:
        prompt = build_prompt(entry)
        output = query_model(prompt)

        results.append({
            "id": str(uuid.uuid4()),
            "topic": entry["topic"],
            "url": entry["url"],
            "gold_label": entry["label"],   # not shown to model
            "prompt": prompt,
            "model_output": output,
            "timestamp": datetime.utcnow().isoformat()
        })

    return results

results = run_evaluation(dataset, sample_size=100)


# Save Results

def save_results(results, path="model_behavior_outputs.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

save_results(results)
