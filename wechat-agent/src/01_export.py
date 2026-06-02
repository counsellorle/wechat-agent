import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR


EXPORT_HELP = """
微信数据导出方式 (二选一):

方式一：使用 PyWxDump (推荐)
  1. pip install pywxdump
  2. 运行: pywxdump
  3. 解密后导出为 JSON，放入 data/raw/

方式二：手动构建 JSON 格式
  按以下格式创建 JSON 文件放入 data/raw/：

  [
    {
      "content": "消息内容",
      "sender": "对方昵称",
      "type": "text",
      "timestamp": 1700000000
    },
    ...
  ]
"""

SAMPLE_FORMAT = [
    {
        "content": "在吗？",
        "sender": "好友张三",
        "type": "text",
        "timestamp": 1700000000,
    },
    {
        "content": "在的，怎么了？",
        "sender": "我",
        "type": "text",
        "timestamp": 1700000010,
    },
]


def create_sample():
    sample_file = RAW_DIR / "sample_format.json"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_FORMAT, f, ensure_ascii=False, indent=2)
        print(f"已创建示例文件: {sample_file}")
    else:
        print(f"示例文件已存在: {sample_file}")


def check_raw_data():
    files = list(RAW_DIR.glob("*.json"))
    if files:
        print(f"发现 {len(files)} 个 JSON 文件:")
        for f in files:
            size = f.stat().st_size
            print(f"  {f.name} ({size / 1024:.1f} KB)")
        total = sum(len(json.load(open(f, encoding="utf-8"))) for f in files)
        print(f"共 {total} 条消息")
    else:
        print(f"data/raw/ 中暂无数据文件")
        print(EXPORT_HELP)
        create_sample()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 微信数据导出 ===\n")
    check_raw_data()


if __name__ == "__main__":
    main()
