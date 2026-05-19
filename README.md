# WorldEvals

## Description

This repo replicates and improves the results of the VLM judge in the [WorldGym paper](https://arxiv.org/abs/2506.00613). 

I finetuned Qwen3-VL-8b with TRL to more effectively classify whether the videos in the [RT-1 dataset](https://robotics-transformer1.github.io/) succesfully completed the task action. Intsead of passing in the full video, I extracted the first, last, and two middle frames of the video to send the model 4 different images. The total training set has 3,319 samples and the test set has 835 samples.

I did a few small training runs tuning batch size and learning rate. The differences between those runs were minor. The main thing that actually mattered was scaling up the dataset size up from ~2k samples -> ~4k samples.

The finetuned model can be found on Hugging Face at: [loganbolton/RT-1-Success-Predictor-qwen3-vl-8b](https://huggingface.co/loganbolton/RT-1-Success-Predictor-qwen3-vl-8b)

## Eval Results (n=835)

Finetuned results averaged over 2 eval runs.

| Model | Accuracy | TPR | TNR | FPR | FNR |
|---|---|---|---|---|---|
| GPT-4o (2024-11-20) | 65.9% | **85.3%** | 52.0% | 48.0% | 14.7% |
| Qwen3-VL-8B (base) | 63.1% | 84.5% | 47.8% | 52.2% | 15.5% |
| Qwen3-VL-8B (finetuned) | **87.1%** | 81.2% | **91.3%** | **8.7%** | **18.9%** |


![Eval Results](assets/eval_results.png)