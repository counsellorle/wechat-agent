import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, TRAIN_DIR

PERSONA_TEMPLATE = """你是一个真实的人类，正在和好友聊天。
你的说话风格：
- 使用自然的口语化表达
- 语气亲切友好
- 回复长度适中，不会长篇大论
- 会使用常见的网络用语

请用自然的方式回复，不要像AI一样说话。"""


def load_processed(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_qapairs(conversations: list, context_window: int = 5) -> list:
    samples = []
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
            sample = {
                "system": PERSONA_TEMPLATE,
                "user": user_content,
                "context": "\n".join(history),
                "assistant": msg["content"],
            }
            samples.append(sample)
    return samples


def save_dataset(samples: list, output_dir: Path, split_ratio: float = 0.8):
    split = int(len(samples) * split_ratio)
    train = samples[:split]
    eval = samples[split:]
    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"
    for file, data in [(train_file, train), (eval_file, eval)]:
        with open(file, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"训练集: {len(train)} 条 -> {train_file}")
    print(f"验证集: {len(eval)} 条 -> {eval_file}")


def main():
    processed_file = PROCESSED_DIR / "cleaned.json"
    if not processed_file.exists():
        print(f"错误: 未找到 {processed_file}")
        return
    print("加载清洗数据...")
    conversations = load_processed(str(processed_file))
    print(f"提取问答对...")
    samples = extract_qapairs(conversations)
    print(f"共提取 {len(samples)} 条训练样本")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, TRAIN_DIR)


if __name__ == "__main__":
    main()
