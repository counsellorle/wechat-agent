import json
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR


def load_raw_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_system_msg(msg: dict) -> bool:
    content = msg.get("content", "")
    if re.match(r"^(系统消息|你已添加|对方已|你已|.* withdrew)", content):
        return True
    return False


def is_noise(msg: dict) -> bool:
    content = msg.get("content", "").strip()
    if not content:
        return True
    if re.match(r"^[\[\(].*[\]\)]$", content):
        return True
    noise_patterns = [
        r"^<.*>$",
        r"^https?://",
        r"^@",
        r"^//",
    ]
    return any(re.match(p, content) for p in noise_patterns)


def is_non_text(msg: dict) -> bool:
    msg_type = msg.get("type", "")
    return msg_type in ("image", "video", "voice", "emoji", "file", "location")


def clean_messages(raw_msgs: list) -> list:
    cleaned = []
    for msg in raw_msgs:
        if is_system_msg(msg) or is_non_text(msg) or is_noise(msg):
            continue
        cleaned.append(msg)
    return cleaned


def merge_short_messages(messages: list, max_gap_seconds: int = 120) -> list:
    merged = []
    for msg in messages:
        if not merged:
            merged.append(msg)
            continue
        last = merged[-1]
        same_sender = msg.get("sender") == last.get("sender")
        if same_sender and is_short(msg.get("content", "")):
            last["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)
    return merged


def is_short(text: str, max_len: int = 10) -> bool:
    return len(text) <= max_len


def parse_timestamp(ts: str) -> float:
    from datetime import datetime
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
    except:
        return 0


def split_conversations(messages: list, max_gap_minutes: int = 30) -> list:
    if not messages:
        return []
    convs = []
    current = [messages[0]]
    for i in range(1, len(messages)):
        t1 = parse_timestamp(messages[i - 1].get("timestamp", ""))
        t2 = parse_timestamp(messages[i].get("timestamp", ""))
        if t2 - t1 > max_gap_minutes * 60:
            convs.append(current)
            current = []
        current.append(messages[i])
    if current:
        convs.append(current)
    return convs


def normalize_role(messages: list, my_name: str) -> list:
    for msg in messages:
        sender = msg.get("sender", "")
        if sender == my_name or sender == "自己":
            msg["role"] = "assistant"
        else:
            msg["role"] = "user"
    return messages


def save_processed(conversations: list, output_path: str):
    data = []
    for conv in conversations:
        data.append({
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in conv
            ]
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"未在 {RAW_DIR} 中找到 JSON 文件")
        return
    my_name = input("请输入你的微信昵称: ")
    all_convs = []
    for rf in raw_files:
        print(f"处理: {rf.name}")
        raw = load_raw_data(str(rf))
        cleaned = clean_messages(raw)
        cleaned = merge_short_messages(cleaned)
        cleaned = normalize_role(cleaned, my_name)
        convs = split_conversations(cleaned)
        all_convs.extend(convs)
        print(f"  {len(raw)} -> {len(cleaned)} 条消息, {len(convs)} 个对话段")
    output = PROCESSED_DIR / "cleaned.json"
    save_processed(all_convs, str(output))
    print(f"已保存 {len(all_convs)} 个对话段到 {output}")


if __name__ == "__main__":
    main()
