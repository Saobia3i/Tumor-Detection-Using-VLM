import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_id: str
    dataset_name: str
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning scaffold for Qwen2-VL-2B on Kaggle GPU"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/qwen2vl-qlora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()

    return TrainingConfig(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def main() -> None:
    cfg = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(cfg.dataset_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=10,
        bf16=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
