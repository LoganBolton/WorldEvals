import argparse
import io
import json
import pickle
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


HF_BASE_URL = (
    "https://huggingface.co/datasets/jxu124/OpenX-Embodiment/resolve/main/"
    "fractal20220817_data/fractal20220817_data_{shard:05d}.tar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream RT-1/fractal20220817_data episodes from the Hugging Face "
            "OpenX-Embodiment mirror and export videos for VLM reward evals."
        )
    )
    parser.add_argument("--output-dir", default="rt1_eval")
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument(
        "--target-successes",
        type=int,
        help="Override --target-per-class for successful episodes.",
    )
    parser.add_argument(
        "--target-failures",
        type=int,
        help="Override --target-per-class for failed episodes.",
    )
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-total-frames",
        type=int,
        help=(
            "Stop after saving approximately this many video frames/images. "
            "An episode is only saved if it fits within the remaining budget."
        ),
    )
    parser.add_argument(
        "--frames-dir",
        help=(
            "Optional directory for JPEG frame exports. If omitted, only MP4 "
            "videos and metadata JSON are written."
        ),
    )
    parser.add_argument(
        "--hf-base-url",
        default=HF_BASE_URL,
        help="URL template with a {shard} placeholder.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Resume from existing data. Reads the manifest in --output-dir, "
            "keeps existing episodes, and continues numbering from where it "
            "left off. Existing success/failure counts are subtracted from the "
            "targets so only the remaining episodes are downloaded."
        ),
    )
    return parser.parse_args()


def decode_image(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def episode_to_frames(episode: dict) -> np.ndarray:
    return np.stack(
        [decode_image(step["observation"]["image"]) for step in episode["steps"]]
    )


def instruction_for(episode: dict) -> str:
    instruction = episode["steps"][0]["observation"]["natural_language_instruction"]
    if isinstance(instruction, bytes):
        return instruction.decode("utf-8")
    return str(instruction)


def bool_value(value) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def aspect_metadata(episode: dict) -> dict:
    return {
        key: bool_value(value)
        for key, value in episode.get("aspects", {}).items()
    }


def episode_is_success(episode: dict) -> bool:
    """Determine success from step rewards (aspects field is unreliable)."""
    rewards = [float(step["reward"]) for step in episode["steps"]]
    return max(rewards) > 0


def save_episode(
    episode: dict,
    output_dir: Path,
    frames_dir: Path | None,
    index: int,
    class_index: int,
    success: bool,
    fps: int,
) -> None:
    stem = f"{index:04d}_{'success' if success else 'fail'}"
    video_path = output_dir / f"{stem}.mp4"
    json_path = output_dir / f"{stem}.json"

    frames = episode_to_frames(episode)
    iio.imwrite(video_path, frames, fps=fps)
    frame_paths = []
    if frames_dir is not None:
        episode_frames_dir = frames_dir / stem
        episode_frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(frames):
            frame_path = episode_frames_dir / f"{frame_index:04d}.jpg"
            Image.fromarray(frame).save(frame_path, format="JPEG", quality=90)
            frame_paths.append(str(frame_path.relative_to(output_dir.parent)))

    json_path.write_text(
        json.dumps(
            {
                "video": video_path.name,
                "frames_dir": (
                    str((frames_dir / stem).relative_to(output_dir.parent))
                    if frames_dir is not None
                    else None
                ),
                "frame_paths": frame_paths,
                "instruction": instruction_for(episode),
                "success": success,
                "aspects": aspect_metadata(episode),
                "max_reward": max(float(step["reward"]) for step in episode["steps"]),
                "class_index": class_index,
                "num_frames": int(frames.shape[0]),
                "fps": fps,
            },
            indent=2,
        )
        + "\n"
    )


def iter_episodes_from_shard(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "WorldEvals/rt1"})
    with urllib.request.urlopen(request) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".data.pickle"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield pickle.load(extracted)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        True: (
            args.target_successes
            if args.target_successes is not None
            else args.target_per_class
        ),
        False: (
            args.target_failures
            if args.target_failures is not None
            else args.target_per_class
        ),
    }
    counts = {True: 0, False: 0}
    frame_count = 0
    total_saved = 0
    manifest = []

    if args.append:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest:
                if entry["success"]:
                    counts[True] += 1
                else:
                    counts[False] += 1
                frame_count += entry.get("num_frames", 0)
            total_saved = len(manifest)
            print(
                f"appending: loaded {total_saved} existing episodes "
                f"(success={counts[True]} fail={counts[False]})",
                flush=True,
            )

    for shard in range(args.start_shard, args.start_shard + args.max_shards):
        url = args.hf_base_url.format(shard=shard)
        print(f"streaming shard {shard:05d}: {url}", flush=True)

        try:
            episodes = iter_episodes_from_shard(url)
            for episode in episodes:
                success = episode_is_success(episode)
                if counts[success] >= targets[success]:
                    continue

                frames = episode["steps"]
                num_frames = len(frames)
                if args.max_total_frames is not None:
                    if num_frames > args.max_total_frames:
                        continue
                    if frame_count + num_frames > args.max_total_frames:
                        print(
                            "frame budget reached: "
                            f"{frame_count}/{args.max_total_frames} frames, "
                            f"skipping {num_frames}-frame episode",
                            flush=True,
                        )
                        (output_dir / "manifest.json").write_text(
                            json.dumps(manifest, indent=2) + "\n"
                        )
                        print(
                            f"done: wrote {total_saved} episodes and {frame_count} frames "
                            f"to {output_dir}"
                        )
                        return

                save_episode(
                    episode=episode,
                    output_dir=output_dir,
                    frames_dir=frames_dir,
                    index=total_saved,
                    class_index=counts[success],
                    success=success,
                    fps=args.fps,
                )
                frame_count += num_frames
                manifest.append(
                    {
                        "video": f"{total_saved:04d}_{'success' if success else 'fail'}.mp4",
                        "json": f"{total_saved:04d}_{'success' if success else 'fail'}.json",
                        "success": success,
                        "num_frames": num_frames,
                    }
                )
                counts[success] += 1
                total_saved += 1
                print(
                    f"saved {total_saved}: success={counts[True]} fail={counts[False]} "
                    f"frames={frame_count}",
                    flush=True,
                )

                if all(counts[label] >= target for label, target in targets.items()):
                    (output_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2) + "\n"
                    )
                    print(f"done: wrote {total_saved} episodes to {output_dir}")
                    return
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            print(f"stopping: shard {shard:05d} was not found", flush=True)
            break

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "ran out of shards before reaching target: "
        f"success={counts[True]}/{targets[True]} fail={counts[False]}/{targets[False]}"
    )
    print(
        f"done: wrote {total_saved} episodes and {frame_count} frames to {output_dir}"
    )


if __name__ == "__main__":
    main()
