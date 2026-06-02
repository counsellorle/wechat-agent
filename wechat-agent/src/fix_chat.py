import json
import re
from pathlib import Path

def parse_wechat_chat(content: str) -> list:
    messages = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if match:
            timestamp = match.group(1)
            remaining = line[len(match.group(1)):].strip()

            sender = "自己"
            msg_content = ""

            if remaining:
                if remaining.startswith("wxid_"):
                    parts = remaining.split("\n", 1)
                    sender = parts[0]
                    msg_content = parts[1] if len(parts) > 1 else ""
                else:
                    msg_content = remaining

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r"\d{4}-\d{2}-\d{2}", next_line) and not next_line.startswith("wxid_"):
                    msg_content = next_line
                    i += 1

            if msg_content:
                messages.append({
                    "timestamp": timestamp,
                    "sender": sender,
                    "content": msg_content
                })
        i += 1
    return messages

def fix_chat_file(input_path: str, output_path: str = None):
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if isinstance(content, str):
        messages = parse_wechat_chat(content)
    else:
        messages = content

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    print(f"已修复: {len(messages)} 条消息 -> {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fix_chat_file(sys.argv[1])
    else:
        fix_chat_file("F:\\Ai\\wechat-agent\\data\\raw\\嘉怡.json")