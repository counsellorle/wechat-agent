import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_MODEL_NAME, BASE_MODEL_PATH, BASE_MODEL_DIR, LORA_DIR, TRAIN_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH, LEARNING_RATE, SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS,
    USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT,
)

import torch
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset, DatasetDict
from trl import SFTTrainer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_dataset():
    train_file = TRAIN_DIR / "train.jsonl"
    eval_file = TRAIN_DIR / "eval.jsonl"
    if not train_file.exists():
        print(f"错误: 未找到训练数据 {train_file}")
        print("请先运行 04_build_dataset.py")
        sys.exit(1)
    train_data = [json.loads(l) for l in open(train_file, encoding="utf-8")]
    eval_data = []
    if eval_file.exists():
        eval_data = [json.loads(l) for l in open(eval_file, encoding="utf-8")]
    return train_data, eval_data


def format_sample(sample: dict) -> str:
    parts = [f"<|im_start|>system\n{sample['system']}"]
    if sample.get("context"):
        parts.append(sample["context"])
    parts.append(f"<|im_start|>user\n{sample['user']}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{sample['assistant']}<|im_end|>")
    return "\n".join(parts)


def main():
    print("加载数据集...")
    train_data, eval_data = load_dataset()
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    train_texts = [format_sample(s) for s in train_data]
    eval_texts = [format_sample(s) for s in eval_data] if eval_data else None

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE_MAP[BNB_4BIT_COMPUTE_DTYPE],
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )

    print(f"加载基座模型 {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=DTYPE_MAP[BNB_4BIT_COMPUTE_DTYPE],
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    LORA_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(LORA_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps" if eval_data else "no",
        learning_rate=LEARNING_RATE,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=2,
    )

    train_dataset = HFDataset.from_list([{"text": t} for t in train_texts])
    eval_dataset = HFDataset.from_list([{"text": t} for t in eval_texts]) if eval_texts else None

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    print("开始训练...")
    trainer.train()

    print(f"保存模型到 {LORA_DIR}")
    trainer.save_model(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print("训练完成")


if __name__ == "__main__":
    main()
