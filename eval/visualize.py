"""Generate a self-contained HTML report for visual inspection of eval results."""

import argparse
import base64
import io
import json
from pathlib import Path

from datasets import load_from_disk


def encode_frame(image) -> str:
    """Encode a PIL image as a base64 JPEG data URI."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_html(results: list[dict], summary: dict | None, frame_uris: dict[int, list[str]]) -> str:
    """Build the full HTML report string."""
    # Summary section
    if summary:
        summary_html = "<div class='summary'><h2>Summary</h2><table>"
        for k, v in summary.items():
            summary_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        summary_html += "</table></div>"
    else:
        summary_html = ""

    # Build cards
    cards_html = ""
    for r in results:
        idx = r["index"]
        status = "correct" if r["correct"] is True else ("incorrect" if r["correct"] is False else "unknown")
        badge_text = status.capitalize()

        frames_html = ""
        if idx in frame_uris:
            for i, uri in enumerate(frame_uris[idx]):
                frames_html += f"<img src='{uri}' alt='frame_{i}' />"

        raw_output = r.get("raw_output", "")

        cards_html += f"""
        <div class='card {status}' data-status='{status}'>
            <div class='card-header'>
                <span class='index'>#{idx}</span>
                <span class='badge badge-{status}'>{badge_text}</span>
            </div>
            <div class='instruction'><strong>Instruction:</strong> {r['instruction']}</div>
            <div class='frames'>{frames_html}</div>
            <div class='labels'>
                <span><strong>Ground Truth:</strong> {r['ground_truth']}</span>
                <span><strong>Prediction:</strong> {r['parsed_answer']}</span>
            </div>
            <details>
                <summary>Raw Model Output</summary>
                <pre>{raw_output}</pre>
            </details>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Results Report</title>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5;
}}
h1 {{ margin-bottom: 10px; }}
.summary {{
    background: white; border-radius: 8px; padding: 16px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.summary table {{ border-collapse: collapse; }}
.summary td {{ padding: 4px 12px 4px 0; }}
.filters {{
    margin-bottom: 20px;
}}
.filters button {{
    padding: 8px 16px; margin-right: 8px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 14px; background: #ddd;
}}
.filters button.active {{ background: #333; color: white; }}
.card {{
    background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border-left: 5px solid #ccc;
}}
.card.correct {{ border-left-color: #2ecc71; }}
.card.incorrect {{ border-left-color: #e74c3c; }}
.card.unknown {{ border-left-color: #f39c12; }}
.card.hidden {{ display: none; }}
.card-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
}}
.index {{ font-weight: bold; font-size: 16px; }}
.badge {{
    padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white;
}}
.badge-correct {{ background: #2ecc71; }}
.badge-incorrect {{ background: #e74c3c; }}
.badge-unknown {{ background: #f39c12; }}
.instruction {{ margin-bottom: 10px; }}
.frames {{
    display: flex; gap: 8px; margin-bottom: 10px; overflow-x: auto;
}}
.frames img {{
    height: 160px; border-radius: 4px; border: 1px solid #eee;
}}
.labels {{
    display: flex; gap: 24px; margin-bottom: 8px;
}}
details {{
    margin-top: 8px;
}}
details pre {{
    background: #f8f8f8; padding: 10px; border-radius: 4px;
    white-space: pre-wrap; word-break: break-word; font-size: 13px;
}}
</style>
</head>
<body>
<h1>Eval Results Report</h1>
{summary_html}
<div class="filters">
    <button class="active" onclick="filterCards('all')">All</button>
    <button onclick="filterCards('correct')">Correct</button>
    <button onclick="filterCards('incorrect')">Incorrect</button>
    <button onclick="filterCards('unknown')">Unknown</button>
</div>
<div id="cards">
{cards_html}
</div>
<script>
function filterCards(status) {{
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.card').forEach(card => {{
        if (status === 'all' || card.dataset.status === status) {{
            card.classList.remove('hidden');
        }} else {{
            card.classList.add('hidden');
        }}
    }});
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization of eval results")
    parser.add_argument("--results-dir", default="eval/results", help="Directory with results.jsonl and summary.json")
    parser.add_argument("--dataset", default="rt1_data/dataset_cache", help="Path to HuggingFace dataset cache")
    parser.add_argument("--output", default="eval/results/report.html", help="Output HTML file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"

    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} results from {results_path}")

    # Load summary if available
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Load dataset to get frames
    print(f"Loading dataset from {args.dataset}...")
    ds = load_from_disk(args.dataset)
    test_ds = ds["test"]

    # Encode frames for each result
    indices = [r["index"] for r in results]
    frame_uris: dict[int, list[str]] = {}
    for i, idx in enumerate(indices):
        sample = test_ds[idx]
        uris = []
        for frame_key in ["frame_0", "frame_1", "frame_2", "frame_3"]:
            uris.append(encode_frame(sample[frame_key]))
        frame_uris[idx] = uris
        if (i + 1) % 10 == 0 or (i + 1) == len(indices):
            print(f"  Encoded frames for {i + 1}/{len(indices)} samples")

    # Generate HTML
    html = build_html(results, summary, frame_uris)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Report written to {output_path} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
