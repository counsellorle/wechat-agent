import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, TRAIN_DIR, LORA_DIR, BASE_MODEL_NAME, BASE_MODEL_PATH, BASE_MODEL_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH, LEARNING_RATE, SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS,
    USE_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT,
    RETRAIN_THRESHOLD,
)
from src.memory import ChatMemory

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from trl import SFTTrainer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
STATE_FILE = Path(__file__).parent.parent / "db" / "retrain_state.json"


def load_state() -> int:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f).get("last_count", 0)
    return 0


def save_state(count: int):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_count": count}, f)


def load_original_data():
    processed_file = PROCESSED_DIR / "cleaned.json"
    if not processed_file.exists():
        return []
    with open(processed_file, encoding="utf-8") as f:
        return json.load(f)


def build_persona_prompt() -> str:
    return """你是一个真实的人类，正在和好友聊天。
你的说话风格：
- 使用自然的口语化表达
- 语气亲切友好
- 回复长度适中
- 会使用常见的网络用语

请用自然的方式回复，不要像AI一样说话。"""


def conversations_to_samples(conversations: list, context_window: int = 5) -> list:
    samples = []
    persona = build_persona_prompt()
    for conv in conversations:
        msgs = conv.get("messages", [])
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            context = msgs[max(0, i - context_window):i]
            history = []
            for ctx in context:
                role = "user" if ctx["role"] == "user" else "assistant"
                history.append(f"<|im_start|>{role}\n{ctx['content']}")
            user_content = ""
            if context and context[-1]["role"] == "user":
                user_content = context[-1]["content"]
            samples.append({
                "system": persona,
                "user": user_content,
                "context": "\n".join(history),
                "assistant": msg["content"],
            })
    return samples


def format_sample(sample: dict) -> str:
    parts = [f"<|im_start|>system\n{sample['system']}"]
    if sample.get("context"):
        parts.append(sample["context"])
    parts.append(f"<|im_start|>user\n{sample['user']}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{sample['assistant']}<|im_end|>")
    return "\n".join(parts)


def main():
    print("=== 增量重训脚本 ===\n")

    memory = ChatMemory()
    new_messages = memory.get_all_messages_for_retrain()
    last_count = load_state()
    new_count = len(new_messages)

    added = new_count - last_count
    print(f"历史消息总数: {new_count}, 上次重训时: {last_count}, 新增: {added}")

    if added < RETRAIN_THRESHOLD:
        print(f"新增消息不足 {RETRAIN_THRESHOLD} 条 (当前 {added} 条)，跳过重训")
        return

    print("加载原始微信数据...")
    original = load_original_data()
    print(f"原始对话段: {len(original)}")

    print("将新对话转换为训练格式...")
    chat_convs = [{"messages": [{"role": m[0], "content": m[1]}]} for m in new_messages]
    all_convs = original + chat_convs
    print(f"总对话段: {len(all_convs)}")

    samples = conversations_to_samples(all_convs)
    print(f"总训练样本: {len(samples)}")

    texts = [format_sample(s) for s in samples]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_compute_dtype=DTYPE_MAP[BNB_4BIT_COMPUTE_DTYPE],
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("加载模型...")
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

    training_args = TrainingArguments(
        output_dir=str(LORA_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=2,
    )

    dataset = HFDataset.from_list([{"text": t} for t in texts])
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    print("开始增量训练...")
    trainer.train()

    print(f"保存新权重到 {LORA_DIR}")
    trainer.save_model(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))

    save_state(new_count)
    print("增量重训完成！新模型已替换旧模型。")


if __name__ == "__main__":
    main()
