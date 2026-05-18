# WorldEvals

## Eval Results (n=835)

| Model | Accuracy | TPR | TNR | FPR | FNR |
|---|---|---|---|---|---|
| GPT-4o (2024-11-20) | 65.9% | 85.3% | 52.0% | 48.0% | 14.7% |
| Qwen3-VL-8B (base) | 63.1% | 84.5% | 47.8% | 52.2% | 15.5% |
| Qwen3-VL-8B (finetuned) | **87.5%** | **85.3%** | **92.0%** | **8.0%** | **14.7%** |

| Finetuned delta | vs GPT-4o | vs Base |
|---|---|---|
| Accuracy | +21.6% | +24.4% |
| TPR | +0.0% | +0.8% |
| TNR | +40.0% | +44.2% |

![Eval Results](finetune/outputs/eval_results.png)