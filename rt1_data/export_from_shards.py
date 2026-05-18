"""Export RT-1 array_record shards to inspectable MP4 videos + JSON metadata.

Usage:
    python export_from_shards.py                          # all shards, default output
    python export_from_shards.py --max-episodes 50        # first 50 episodes
    python export_from_shards.py --output-dir my_export   # custom output dir
    python export_from_shards.py --failures-only          # only failed episodes
"""

import argparse
import glob
import io
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordReader
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RT-1 array_record shards to MP4 + JSON."
    )
    parser.add_argument(
        "--shard-dir",
        default=str(Path(__file__).parent / "shards"),
        help="Directory containing fractal_fractal_*.array_record-* files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "exported"),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--successes-only", action="store_true")
    return parser.parse_args()


def extract_episode(example: tf.train.Example) -> dict:
    feat = example.features.feature

    num_steps = feat["steps/num_steps"].int64_list.value[0]
    instruction = feat[
        "steps/observation/natural_language_instruction"
    ].bytes_list.value[0].decode()
    rewards = list(feat["steps/reward"].float_list.value)
    success = max(rewards) > 0
    jpeg_bytes_list = list(
        feat["steps/observation/image"].bytes_list.value
    )

    frames = []
    for jpeg_bytes in jpeg_bytes_list:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            frames.append(np.asarray(img.convert("RGB")))

    return {
        "frames": np.stack(frames),
        "instruction": instruction,
        "success": success,
        "num_steps": num_steps,
    }


def main() -> None:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    args = parse_args()
    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(shard_dir / "fractal_fractal_*")))
    if not files:
        print(f"No shard files found in {shard_dir}")
        return

    print(f"Found {len(files)} shards in {shard_dir}")

    total_saved = 0
    total_scanned = 0
    manifest = []

    for fpath in files:
        reader = ArrayRecordReader(fpath)
        n = reader.num_records()
        shard_name = Path(fpath).name[-20:]

        for ep_idx in range(n):
            if args.max_episodes is not None and total_saved >= args.max_episodes:
                reader.close()
                break

            raw = reader.read([ep_idx])[0]
            example = tf.train.Example()
            example.ParseFromString(raw)
            ep = extract_episode(example)
            total_scanned += 1

            if args.failures_only and ep["success"]:
                continue
            if args.successes_only and not ep["success"]:
                continue

            tag = "success" if ep["success"] else "fail"
            stem = f"{total_saved:04d}_{tag}"
            video_path = output_dir / f"{stem}.mp4"
            json_path = output_dir / f"{stem}.json"

            iio.imwrite(video_path, ep["frames"], fps=args.fps)
            json_path.write_text(
                json.dumps(
                    {
                        "video": video_path.name,
                        "instruction": ep["instruction"],
                        "success": ep["success"],
                        "num_frames": int(ep["frames"].shape[0]),
                        "fps": args.fps,
                    },
                    indent=2,
                )
                + "\n"
            )

            manifest.append(
                {
                    "video": f"{stem}.mp4",
                    "json": f"{stem}.json",
                    "instruction": ep["instruction"],
                    "success": ep["success"],
                    "num_frames": int(ep["frames"].shape[0]),
                }
            )
            total_saved += 1
            if total_saved % 50 == 0:
                s = sum(1 for m in manifest if m["success"])
                f = total_saved - s
                print(
                    f"  saved {total_saved} (success={s}, fail={f}), "
                    f"scanned {total_scanned}",
                    flush=True,
                )
        else:
            reader.close()
            continue
        break

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    s = sum(1 for m in manifest if m["success"])
    f = total_saved - s
    print(
        f"Done: {total_saved} episodes (success={s}, fail={f}) "
        f"from {total_scanned} scanned -> {output_dir}"
    )


if __name__ == "__main__":
    main()
