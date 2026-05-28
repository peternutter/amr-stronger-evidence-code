"""
Utility functions for inspecting JSONL results from MASK benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import HTML, Markdown, display


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts"""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_summary_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Get summary statistics from results"""
    stats: dict[str, Any] = {
        "total": len(results),
        "splits": {},
        "types": {},
        "total_pressure_responses": 0,
        "total_belief_responses": 0,
    }

    for r in results:
        # Count by split and type
        split: str = r["split"]
        type_: str = r["type"]
        splits_dict: dict[str, int] = stats["splits"]
        types_dict: dict[str, int] = stats["types"]
        splits_dict[split] = splits_dict.get(split, 0) + 1
        types_dict[type_] = types_dict.get(type_, 0) + 1

        # Count responses
        stats["total_pressure_responses"] += len(r["pressure_responses"])
        stats["total_belief_responses"] += sum(len(b["responses"]) for b in r["belief_responses"].values())

    return stats


def print_summary(results: list[dict[str, Any]]):
    """Print summary statistics"""
    stats = get_summary_stats(results)

    print(f"Total examples: {stats['total']}")

    print("\nBy split:")
    for split, count in sorted(stats["splits"].items()):
        print(f"  {split}: {count}")

    print("\nBy type:")
    for type_, count in sorted(stats["types"].items()):
        print(f"  {type_}: {count}")

    print("\nResponses collected:")
    print(f"  Pressure responses: {stats['total_pressure_responses']}")
    print(f"  Belief responses: {stats['total_belief_responses']}")
    total = stats["total_pressure_responses"] + stats["total_belief_responses"]
    print(f"  Total: {total}")


def format_message_html(msg: dict[str, str]) -> str:
    """Format a single message as HTML with color coding"""
    role = msg["role"]
    content = msg["content"]

    role_colors = {
        "system": "#e3f2fd",
        "user": "#f3e5f5",
        "assistant": "#e8f5e9",
    }

    bg_color = role_colors.get(role, "#f5f5f5")

    return f"""
    <div style="margin: 10px 0; padding: 10px; background-color: {bg_color};
                border-radius: 5px; border-left: 4px solid #666;">
        <strong style="text-transform: uppercase; color: #333;">[{role}]</strong><br/>
        <div style="margin-top: 5px; white-space: pre-wrap;
                    font-family: 'Courier New', monospace; line-height: 1.5;">
            {content}
        </div>
    </div>
    """


def display_example(example: dict[str, Any], index: int | None = None):
    """Display a single example with nice formatting in Jupyter"""
    if index is not None:
        display(Markdown(f"## Example {index}"))

    # Metadata
    display(Markdown("### Metadata"))
    metadata_df = pd.DataFrame(
        [
            {
                "Task ID": example["task_id"],
                "Split": example["split"],
                "Type": example["type"],
                "Ground Truth": example["ground_truth"],
            }
        ]
    )
    display(metadata_df)

    # Proposition
    display(Markdown("**Proposition:**"))
    display(Markdown(f"> {example['proposition']}"))

    # Pressure prompt
    display(Markdown("### Pressure Prompt (Lie Elicitation)"))
    html = "".join([format_message_html(msg) for msg in example["pressure_prompt"]])
    display(HTML(html))

    # Pressure responses
    num_pressure = len(example["pressure_responses"])
    display(Markdown(f"**Pressure Responses** ({num_pressure} runs):"))
    for resp in example["pressure_responses"]:
        display(Markdown(f"- **Run {resp['run']}:** {resp['response']}"))

    # Belief responses
    for belief_key, belief_data in example["belief_responses"].items():
        display(Markdown(f"### {belief_key.replace('_', ' ').title()}"))
        html = "".join([format_message_html(msg) for msg in belief_data["prompt"]])
        display(HTML(html))

        num_resps = len(belief_data["responses"])
        display(Markdown(f"**Responses** ({num_resps} runs):"))
        for resp in belief_data["responses"]:
            display(Markdown(f"- **Run {resp['run']}:** {resp['response']}"))


def print_prompts_only(example: dict[str, Any], index: int | None = None):
    """Print just the prompts (no responses) for an example"""
    if index is not None:
        print("=" * 80)
        print(f"PROMPTS FOR EXAMPLE {index}")
        print("=" * 80)

    print("\n[PRESSURE PROMPT]")
    for msg in example["pressure_prompt"]:
        print(f"\n{msg['role'].upper()}:")
        print(msg["content"])

    print("\n\n[BELIEF PROMPTS]")
    for belief_key, belief_data in example["belief_responses"].items():
        print(f"\n--- {belief_key} ---")
        for msg in belief_data["prompt"]:
            print(f"\n{msg['role'].upper()}:")
            print(msg["content"])
