import argparse
import base64
import json
import re
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from openai import OpenAI
from PIL import Image


FINAL_SCORE_RE = re.compile(r"Final\s*Score:\s*(0(?:\.0)?|1(?:\.0)?)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score RT-1 rollout videos with a VLM.")
    parser.add_argument("--input-dir", default="rt1_eval")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="vlm_results.jsonl")
    return parser.parse_args()


def load_examples(input_dir: Path):
    for json_path in sorted(input_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        metadata = json.loads(json_path.read_text())
        video_path = input_dir / metadata["video"]
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        yield json_path, video_path, metadata


def sampled_frames(video_path: Path, max_frames: int) -> list[np.ndarray]:
    frames = list(iio.imiter(video_path))
    if len(frames) <= max_frames:
        return frames
    indexes = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in indexes]


def frame_data_url(frame: np.ndarray) -> str:
    image = Image.fromarray(frame).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(instruction: str) -> str:
    return f"""
Here is a sequence of frames from a robot policy rollout video.
I need your help determining whether the policy is successful. How successfully does the robot complete the following task?

Instruction: {instruction}
Score rubric:
0 = Failure: instruction "{instruction}" not completed.
1 = Success: instruction completed.

Provide brief reasoning in 2-3 sentences. Then output EXACTLY one final line:
Final Score: X
Where X is 0 or 1.
No extra numbers after that line.
""".strip()


def parse_score(text: str) -> int:
    match = FINAL_SCORE_RE.search(text)
    if not match:
        raise ValueError(f"could not parse final score from response: {text!r}")
    return int(float(match.group(1)))


def score_video(
    client: OpenAI,
    model: str,
    video_path: Path,
    instruction: str,
    max_frames: int,
    detail: str,
) -> tuple[int, str]:
    content = [{"type": "input_text", "text": build_prompt(instruction)}]
    for frame in sampled_frames(video_path, max_frames=max_frames):
        content.append(
            {
                "type": "input_image",
                "image_url": frame_data_url(frame),
                "detail": detail,
            }
        )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )
    text = response.output_text
    return parse_score(text), text


def metrics(results: list[dict]) -> dict:
    tp = sum(r["label"] and r["prediction"] for r in results)
    fp = sum((not r["label"]) and r["prediction"] for r in results)
    fn = sum(r["label"] and (not r["prediction"]) for r in results)
    tn = sum((not r["label"]) and (not r["prediction"]) for r in results)

    positives = tp + fn
    negatives = tn + fp
    return {
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
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    client = OpenAI()

    results = []
    with output_path.open("w") as out:
        for index, (json_path, video_path, metadata) in enumerate(load_examples(input_dir)):
            if args.limit is not None and index >= args.limit:
                break

            prediction, raw_response = score_video(
                client=client,
                model=args.model,
                video_path=video_path,
                instruction=metadata["instruction"],
                max_frames=args.max_frames,
                detail=args.detail,
            )
            result = {
                "metadata": json_path.name,
                "video": video_path.name,
                "instruction": metadata["instruction"],
                "label": bool(metadata["success"]),
                "prediction": bool(prediction),
                "raw_response": raw_response,
            }
            out.write(json.dumps(result) + "\n")
            out.flush()
            results.append(result)
            print(
                f"{video_path.name}: label={result['label']} prediction={result['prediction']}",
                flush=True,
            )

    summary = metrics(results)
    Path(f"{output_path}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
