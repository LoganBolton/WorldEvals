import os
import re
import numpy as np
import torch
import torch._dynamo
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TextStreamer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from prepare_dataset import load_dataset, prepare_train_dataset, prepare_eval_samples

# Fix: Disable torch.compile (dynamo) to avoid recompile limit crash
# Vision models produce variable-length sequences that trigger excessive recompilations
torch._dynamo.config.disable = True


# ---------------------------------------------------------------------------
# Model & LoRA
# ---------------------------------------------------------------------------

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    device_map="auto",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

dataset = load_dataset()
train_dataset = prepare_train_dataset(dataset)
eval_samples = prepare_eval_samples(dataset)
print(f"Train: {len(train_dataset)}  |  Eval: {len(eval_samples)}")


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

class EvalCallback(TrainerCallback):
    """Evaluates once per epoch by generating predictions on the full test set."""

    def __init__(self, eval_samples, tokenizer, steps_per_epoch):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.steps_per_epoch = steps_per_epoch
        self.metrics = {"step": [], "epoch": [], "accuracy": [], "tpr": [], "tnr": [], "fpr": [], "fnr": []}
        self.train_log = {"step": [], "loss": []}
        self.best_accuracy = -1.0
        self.best_step = -1
        self.best_epoch = -1

    def _evaluate(self, model, step, epoch):
        was_training = model.training
        model.eval()

        tp = fp = tn = fn = unk = 0
        with torch.no_grad():
            for images, prompt, gt in self.eval_samples:
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}, {"type": "image"},
                    {"type": "image"}, {"type": "image"},
                ]}]
                input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.tokenizer(
                    images, input_text,
                    add_special_tokens=False, return_tensors="pt",
                ).to("cuda")
                out_ids = model.generate(
                    **inputs, max_new_tokens=32, use_cache=True, temperature=1.0,
                )
                text = self.tokenizer.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )
                m = re.search(r"<answer>\s*(Success|Failure)\s*</answer>", text)
                if m:
                    pred = m.group(1)
                    if   pred == "Success" and gt == "Success": tp += 1
                    elif pred == "Success" and gt == "Failure": fp += 1
                    elif pred == "Failure" and gt == "Failure": tn += 1
                    else:                                       fn += 1
                else:
                    unk += 1

        total = tp + fp + tn + fn + unk
        acc = (tp + tn) / total if total else 0
        tpr = tp / (tp + fn) if (tp + fn) else 0
        tnr = tn / (tn + fp) if (tn + fp) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0

        self.metrics["step"].append(step)
        self.metrics["epoch"].append(epoch)
        self.metrics["accuracy"].append(acc)
        self.metrics["tpr"].append(tpr)
        self.metrics["tnr"].append(tnr)
        self.metrics["fpr"].append(fpr)
        self.metrics["fnr"].append(fnr)

        tag = ""
        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_step = step
            self.best_epoch = epoch
            tag = " ★ new best"
            model.save_pretrained("outputs/best_checkpoint")
            self.tokenizer.save_pretrained("outputs/best_checkpoint")

        print(f"  [Epoch {epoch:.1f} / step {step}] Acc={acc:.1%}  TPR={tpr:.1%}  TNR={tnr:.1%}  |  TP:{tp} FP:{fp} TN:{tn} FN:{fn} Unk:{unk}{tag}")

        if was_training:
            model.train()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_log["step"].append(state.global_step)
            self.train_log["loss"].append(logs["loss"])

    def on_train_begin(self, args, state, control, **kwargs):
        print("Running baseline eval (epoch 0)...")
        self._evaluate(kwargs["model"], 0, 0)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch or (state.global_step / self.steps_per_epoch)
        self._evaluate(kwargs["model"], state.global_step, epoch)


steps_per_epoch = 170
eval_callback = EvalCallback(eval_samples, tokenizer, steps_per_epoch)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    callbacks=[eval_callback],
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

trainer_stats = trainer.train()


# ---------------------------------------------------------------------------
# Plot metrics
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training loss
if eval_callback.train_log["step"]:
    axes[0].plot(eval_callback.train_log["step"], eval_callback.train_log["loss"],
                 "b-", alpha=0.6, linewidth=1)
    window = min(10, len(eval_callback.train_log["loss"]))
    if window > 1:
        loss_arr = np.array(eval_callback.train_log["loss"])
        smoothed = np.convolve(loss_arr, np.ones(window) / window, mode="valid")
        axes[0].plot(eval_callback.train_log["step"][window - 1:], smoothed,
                     "b-", linewidth=2, label=f"Smoothed (w={window})")
        axes[0].legend()
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

# Accuracy
epochs = eval_callback.metrics["epoch"]
axes[1].plot(epochs, eval_callback.metrics["accuracy"], "g-o", markersize=8, linewidth=2, label="Accuracy")
if eval_callback.best_epoch >= 0:
    axes[1].axvline(x=eval_callback.best_epoch, color="g", linestyle="--", alpha=0.4,
                    label=f"Best: {eval_callback.best_accuracy:.1%} (epoch {eval_callback.best_epoch:.0f})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Eval Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.05)

# TPR / TNR
axes[2].plot(epochs, eval_callback.metrics["tpr"], "b-o", markersize=6, linewidth=2, label="TPR (Sensitivity)")
axes[2].plot(epochs, eval_callback.metrics["tnr"], "r-s", markersize=6, linewidth=2, label="TNR (Specificity)")
axes[2].plot(epochs, eval_callback.metrics["fpr"], "r-^", markersize=5, linewidth=1, alpha=0.5, label="FPR")
axes[2].plot(epochs, eval_callback.metrics["fnr"], "b-v", markersize=5, linewidth=1, alpha=0.5, label="FNR")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Rate")
axes[2].set_title("TPR / TNR / FPR / FNR")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1.05)

fig.suptitle(
    f"Training Performance  —  Best accuracy: {eval_callback.best_accuracy:.1%} at epoch {eval_callback.best_epoch:.0f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("outputs/training_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nBest checkpoint: outputs/best_checkpoint  (epoch {eval_callback.best_epoch:.0f}, step {eval_callback.best_step}, accuracy {eval_callback.best_accuracy:.1%})")
