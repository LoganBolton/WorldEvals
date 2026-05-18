"""Evaluate the base Qwen3-VL-8B model on the RT-1 test set."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from unsloth import FastVisionModel

ANSWER_RE = re.compile(r"<answer>\s*(Success|Failure)\s*</answer>", re.I)

PROMPT_TEMPLATE = """\
Here is a sequence of frames from a robot policy which has been \
rolled out in a video-generation-based world model. I need your help \
determining whether the policy is successful. How successfully does \
the robot complete the following task?

## Task Description: "{instruction}"

## Score rubric:
Failure: instruction "{instruction}" not completed.
Success: instruction completed.

## Answer Format
Place your final answer in <answer></answer> tags. For example if the robot \
completed the task, your answer would be:
<answer>Success</answer>

If the robot did not complete the task, your answer would be:
<answer>Failure</answer>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base Qwen3-VL on RT-1 test set.")
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        help="HuggingFace model ID to load via Unsloth.",
    )
    parser.add_argument(
        "--dataset",
        default="rt1_data/dataset_cache",
        help="Path to the cached HF DatasetDict.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory to write results.jsonl and summary.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N test samples (for quick testing).",
    )
    return parser.parse_args()


def parse_answer(text: str) -> str:
    """Extract Success/Failure from <answer> tags, or return 'unknown'."""
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).capitalize()
    return "unknown"


def run_inference(model, tokenizer, sample: dict) -> str:
    """Run model inference on a single sample and return the raw output text."""
    instruction = sample["instruction"]
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    images = [sample["frame_0"], sample["frame_1"], sample["frame_2"], sample["frame_3"]]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

    # Decode only the generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_metrics(results: list[dict]) -> dict:
    """Compute confusion matrix and rates from results."""
    tp = sum(r["label"] and r["prediction"] for r in results)
    fp = sum((not r["label"]) and r["prediction"] for r in results)
    fn = sum(r["label"] and (not r["prediction"]) for r in results)
    tn = sum((not r["label"]) and (not r["prediction"]) for r in results)

    total = len(results)
    positives = tp + fn
    negatives = tn + fp
    unknown = sum(1 for r in results if r["parsed_answer"] == "unknown")

    return {
        "total": total,
        "unknown": unknown,
        "accuracy": (tp + tn) / total if total else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "true_positive_rate": tp / positives if positives else None,
        "false_negative_rate": fn / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "true_negative_rate": tn / negatives if negatives else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    test_data = dataset["test"]
    n = len(test_data) if args.max_samples is None else min(args.max_samples, len(test_data))
    print(f"Evaluating {n} / {len(test_data)} test samples")

    # Run inference
    results = []
    with results_path.open("w") as out:
        for i in tqdm(range(n), desc="Evaluating"):
            sample = test_data[i]
            raw_output = run_inference(model, tokenizer, sample)
            parsed = parse_answer(raw_output)

            label = bool(sample["success"])
            prediction = parsed == "Success"  # "unknown" maps to False

            ground_truth = "Success" if label else "Failure"
            result = {
                "index": i,
                "instruction": sample["instruction"],
                "ground_truth": ground_truth,
                "parsed_answer": parsed,
                "correct": ground_truth == parsed,
                "label": label,
                "prediction": prediction,
                "raw_output": raw_output,
            }
            out.write(json.dumps(result) + "\n")
            out.flush()
            results.append(result)

            label_str = "Success" if label else "Failure"
            tqdm.write(f"[{i}] label={label_str} pred={parsed} | {sample['instruction']}")

    # Compute and save metrics
    summary = compute_metrics(results)
    summary["model"] = args.model
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
