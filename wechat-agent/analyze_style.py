import json
from collections import Counter

with open(r"F:\Ai\wechat-agent\data\raw\嘉怡.json", "r", encoding="utf-8") as f:
    messages = json.load(f)

# 分离"自己"和对方的消息
my_msgs = [m for m in messages if m["sender"] == "\u81ea\u5df1"]
other_msgs = [m for m in messages if m["sender"] != "\u81ea\u5df1"]

print("=" * 50)
print(f"总消息数: {len(messages)}")
print(f"你的消息数: {len(my_msgs)}")
print(f"对方消息数: {len(other_msgs)}")
print("=" * 50)

print("\n【你发送的所有消息】")
for m in my_msgs:
    print(f"  {m['content']}")

print("\n【对方发送的所有消息】")
for m in other_msgs:
    print(f"  {m['content']}")

# 分析你的说话特点
all_my_text = " ".join(m["content"] for m in my_msgs)

print("\n【你的消息长度统计】")
lengths = [len(m["content"]) for m in my_msgs]
print(f"  平均长度: {sum(lengths)/len(lengths):.1f} 字")
print(f"  最长: {max(lengths)} 字")
print(f"  最短: {min(lengths)} 字")

# 短消息占比（5字以下）
short = sum(1 for l in lengths if l <= 5)
print(f"  短消息(<=5字)占比: {short}/{len(lengths)} = {short/len(lengths)*100:.0f}%")

# 统计常用词
print("\n【常用口语词/语气词】")
keywords = ["哈哈", "嗯", "哦", "吧", "呢", "啊", "啦", "嘛", "呀", "额", "嗯嗯", "诶"]
for kw in keywords:
    count = all_my_text.count(kw)
    if count > 0:
        print(f"  '{kw}': {count}次")

# 统计表情使用
print("\n【表情/符号使用】")
import re
emojis = re.findall(r'$$.*?$$', all_my_text)
for e in emojis:
    print(f"  {e}")

# 分析句子模式
print("\n【句子特征】")
question_count = sum(1 for m in my_msgs if "?" in m["content"] or "？" in m["content"])
print(f"  提问句数: {question_count}")
exclaim_count = sum(1 for m in my_msgs if "!" in m["content"] or "！" in m["content"])
print(f"  感叹句数: {exclaim_count}")
