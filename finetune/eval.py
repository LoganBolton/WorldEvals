import argparse
import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from prepare_dataset import load_dataset, prepare_eval_samples, make_prompt


def encode_image_b64(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Local model inference
# ---------------------------------------------------------------------------

def eval_local(eval_samples, dataset, model_path=None):
    import torch
    import torch._dynamo
    torch._dynamo.config.disable = True
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto",
    )

    if model_path:
        from peft import PeftModel
        print(f"Loading adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

    FastVisionModel.for_inference(model)
    model.eval()

    results = []
    with torch.no_grad():
        for idx, (images, prompt, gt) in enumerate(tqdm(eval_samples, desc="Eval (local)")):
            instruction = dataset["test"][idx]["instruction"]
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image"}, {"type": "image"},
                {"type": "image"}, {"type": "image"},
            ]}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                images, input_text,
                add_special_tokens=False, return_tensors="pt",
            ).to("cuda")
            out_ids = model.generate(
                **inputs, max_new_tokens=32, use_cache=True, temperature=1.0,
            )
            text = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            results.append((text, gt, instruction))

    return results


# ---------------------------------------------------------------------------
# OpenAI API inference
# ---------------------------------------------------------------------------

def eval_openai(eval_samples, dataset, model_id):
    from openai import OpenAI
    client = OpenAI()

    results = []
    for idx, (images, prompt, gt) in enumerate(tqdm(eval_samples, desc=f"Eval ({model_id})")):
        instruction = dataset["test"][idx]["instruction"]
        content = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = encode_image_b64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=128,
        )
        text = response.choices[0].message.content
        results.append((text, gt, instruction))

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results, model_name):
    records = []
    tp = fp = tn = fn = unk = 0
    for text, gt, instruction in results:
        m = re.search(r"<answer>\s*(Success|Failure)\s*</answer>", text)
        pred = m.group(1) if m else None
        if   pred == "Success" and gt == "Success": tp += 1
        elif pred == "Success" and gt == "Failure": fp += 1
        elif pred == "Failure" and gt == "Failure": tn += 1
        elif pred == "Failure" and gt == "Success": fn += 1
        else:                                       unk += 1

        correct = pred == gt if pred else False
        records.append({
            "instruction": instruction,
            "ground_truth": gt,
            "prediction": pred,
            "correct": correct,
            "model_output": text,
        })

    total = tp + fp + tn + fn + unk
    acc = (tp + tn) / total if total else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    tnr = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    print(f"\n{'='*60}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  TPR (Sensitivity): {tpr:.1%}")
    print(f"  TNR (Specificity): {tnr:.1%}")
    print(f"  FPR: {fpr:.1%}  |  FNR: {fnr:.1%}")
    print(f"  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}  Unk:{unk}")
    print(f"{'='*60}")

    # Save to JSON
    output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": acc,
            "tpr": tpr,
            "tnr": tnr,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "unk": unk,
            "total": total,
        },
        "samples": records,
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_name = re.sub(r"[/\\:]", "_", model_name)
    out_path = out_dir / f"eval_{safe_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the RT1 test set")
    parser.add_argument(
        "--model", required=True,
        help="Model to evaluate: 'base' for base Qwen3-VL, "
             "path to a local adapter (e.g. outputs/best_checkpoint), "
             "or an OpenAI model ID (e.g. gpt-4o-2024-11-20)",
    )
    args = parser.parse_args()

    dataset = load_dataset()
    eval_samples = prepare_eval_samples(dataset)
    print(f"Eval samples: {len(eval_samples)}")

    if args.model == "base":
        print("Evaluating base model (no adapter)...")
        results = eval_local(eval_samples, dataset)
    elif args.model.startswith("gpt-") or args.model.startswith("o1") or args.model.startswith("o3") or args.model.startswith("o4"):
        print(f"Evaluating OpenAI model: {args.model}")
        results = eval_openai(eval_samples, dataset, args.model)
    else:
        print(f"Evaluating local adapter: {args.model}")
        results = eval_local(eval_samples, dataset, model_path=args.model)

    score_results(results, args.model)


if __name__ == "__main__":
    main()
