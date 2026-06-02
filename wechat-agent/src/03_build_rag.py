import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from src.rag import RAGRetriever


def load_processed(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_texts(conversations: list) -> list:
    texts = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            text = msg.get("content", "").strip()
            if text:
                texts.append(text)
    return texts


def main():
    processed_file = PROCESSED_DIR / "cleaned.json"
    if not processed_file.exists():
        print(f"错误: 未找到处理后的数据 {processed_file}")
        print("请先运行 02_clean.py")
        return

    print("加载清洗后的数据...")
    conversations = load_processed(str(processed_file))
    texts = extract_texts(conversations)
    print(f"共提取 {len(texts)} 条消息")

    print("初始化 RAG 检索器...")
    retriever = RAGRetriever()
    print(f"当前向量库已有 {retriever.count()} 条记录")

    print("生成向量索引...")
    ids = [f"msg_{i}" for i in range(len(texts))]
    retriever.add_messages(texts, ids)
    print(f"索引完成，当前向量库共 {retriever.count()} 条记录")

    print("\n验证检索：")
    test_query = "你好"
    results = retriever.search(test_query, k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r[:60]}...")


if __name__ == "__main__":
    main()
