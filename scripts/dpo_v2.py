# train_dpo.py
# Minimal DPO training with TRL on a small model + public preference data.

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"          # small + chat-tuned
DATASET = "trl-lib/ultrafeedback_binarized"    # has prompt, chosen, rejected

def main():
    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,   # Qwen uses custom code
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load preference dataset (prompt / chosen / rejected)
    # Tip: during smoke tests, use split="train[:2000]" to iterate faster.
    train_ds = load_dataset(DATASET, split="train[:2000]")
    test_ds = load_dataset(DATASET, split="test[:2000]")

    # DPO training config (keep it simple)
    dpo_cfg = DPOConfig(
        output_dir="Qwen3-4B-dpo",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_steps=200,
        # beta=0.1,                 # DPO inverse temperature
        # max_length=1024,          # prompt + completion truncation
        remove_unused_columns=False,
        report_to=None,           # set to "wandb" or "tensorboard" if you like
        # bf16=True,              # uncomment if your GPU supports bfloat16
        # fp16=True,              # (or use fp16)
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        processing_class=tokenizer,   # tokenizer or processor
        train_dataset=train_ds,
        eval_dataset=test_ds
        # peft_config=...            # see LoRA note below
    )

    trainer.train()
    print("Trained")
    print(trainer)
    trainer.save_model()               # saves to output_dir
    if hasattr(trainer, "processing_class") and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(dpo_cfg.output_dir)

if __name__ == "__main__":
    main()

