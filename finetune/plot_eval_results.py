import numpy as np
import matplotlib.pyplot as plt

models = ["GPT-4o\n(2024-11-20)", "Qwen3-VL-8B\n", "Qwen3-VL-8B\n(finetuned)"]
metrics = ["Accuracy", "TPR", "TNR"]

data = {
    "Accuracy":               [65.9, 63.1, 87.5],
    "TPR":  [85.3, 84.5, 85.3],
    "TNR":  [52.0, 47.8, 92.0],
}

x = np.arange(len(models))
width = 0.25

# Colors per metric: [GPT-4o, Qwen base, Qwen finetuned]
# GPT-4o warmer/lighter, Qwens cooler/deeper — noticeable but not jarring
model_colors = {
    "Accuracy": ["#7BA3D4", "#3B5998", "#3B5998"],
    "TPR":      ["#7CC98A", "#2E8B57", "#2E8B57"],
    "TNR":      ["#E07070", "#993333", "#993333"],
}

fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, data[metric], width, label=metric,
                  color=model_colors[metric])
    for bar, val in zip(bars, data[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_xlabel("Model Evaluated", fontsize=12)
ax.set_title("RT1 Task Success Prediction — Test Metrics (n=835)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("outputs/eval_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to outputs/eval_results.png")
