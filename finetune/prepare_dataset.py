import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Image as HFImage, load_from_disk
from PIL import Image
import imageio.v3 as iio
from tqdm.auto import tqdm

DATASET_CACHE = Path("../rt1_data/dataset_cache")


def sample_frames(video_path: str) -> list:
    """Extract 4 evenly spaced frames: first, two middle, last."""
    frames = iio.imread(video_path, plugin="pyav")
    n = len(frames)
    if n < 4:
        indices = list(range(n))
    else:
        indices = [0, n // 3, 2 * n // 3, n - 1]
    return [Image.fromarray(frames[i]) for i in indices]


def load_rt1_split(split_dir: Path, desc: str = "Loading") -> Dataset:
    records = []
    json_files = sorted(f for f in split_dir.glob("*.json") if f.name != "manifest.json")
    for json_path in tqdm(json_files, desc=desc):
        meta = json.loads(json_path.read_text())
        video_path = str(split_dir / meta["video"])
        imgs = sample_frames(video_path)
        records.append({
            "frame_0": imgs[0],
            "frame_1": imgs[1] if len(imgs) > 1 else imgs[0],
            "frame_2": imgs[2] if len(imgs) > 2 else imgs[-1],
            "frame_3": imgs[3] if len(imgs) > 3 else imgs[-1],
            "instruction": meta["instruction"],
            "success": meta["success"],
        })
    return (
        Dataset.from_list(records)
        .cast_column("frame_0", HFImage())
        .cast_column("frame_1", HFImage())
        .cast_column("frame_2", HFImage())
        .cast_column("frame_3", HFImage())
    )


def load_dataset() -> DatasetDict:
    if DATASET_CACHE.exists():
        print("Loading cached dataset...")
        return load_from_disk(str(DATASET_CACHE))

    print("Extracting frames from videos (one-time)...")
    rt1_root = Path("../rt1_data")
    dataset = DatasetDict({
        "train": load_rt1_split(rt1_root / "train", desc="Train"),
        "test": load_rt1_split(rt1_root / "test", desc="Test"),
    })
    dataset.save_to_disk(str(DATASET_CACHE))
    print(f"Saved to {DATASET_CACHE}")
    return dataset


def make_prompt(instruction: str) -> str:
    return f"""Here is a sequence of frames from a robot policy which has been \
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


def convert_to_conversation(sample):
    instruction = sample["instruction"]
    success = sample["success"]
    prompt = make_prompt(instruction)
    response = "<answer>Success</answer>" if success else "<answer>Failure</answer>"

    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": prompt},
             {"type": "image", "image": sample["frame_0"]},
             {"type": "image", "image": sample["frame_1"]},
             {"type": "image", "image": sample["frame_2"]},
             {"type": "image", "image": sample["frame_3"]},
         ]},
        {"role": "assistant",
         "content": [
             {"type": "text", "text": response}]},
    ]
    return {"messages": conversation}


def prepare_train_dataset(dataset: DatasetDict) -> list:
    return [convert_to_conversation(sample) for sample in dataset["train"]]


def prepare_eval_samples(dataset: DatasetDict) -> list:
    eval_samples = []
    for idx in range(len(dataset["test"])):
        sample = dataset["test"][idx]
        instruction = sample["instruction"]
        ground_truth = "Success" if sample["success"] else "Failure"
        images = [sample["frame_0"], sample["frame_1"], sample["frame_2"], sample["frame_3"]]
        prompt = make_prompt(instruction)
        eval_samples.append((images, prompt, ground_truth))
    return eval_samples


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset)

    train_dataset = prepare_train_dataset(dataset)
    print(f"Train conversations: {len(train_dataset)}")

    eval_samples = prepare_eval_samples(dataset)
    print(f"Eval samples: {len(eval_samples)}")

    print("\nSample conversation:")
    print(train_dataset[0])
