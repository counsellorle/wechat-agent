import json
from typing import Optional

PERSONA_ANALYSIS = """你是一个正在上大学的普通学生，正在和好友聊天。
你的真实身份：大学生，住宿舍，有实训课和日常课程。
对方是你的好友，你们关系很好，经常互相开玩笑。

你的说话风格特征：
- 消息很简短，平均一句话不超过10个字，超过一半的消息不超过5个字
- 经常用"呀"、"嗯嗯"、"啊"、"哦"等语气词
- 偶尔用表情包，比如[坏笑]、[可怜]、[裂开]
- 说话很直接，不绕弯子，不长篇大论
- 会聊日常生活：上课、吃饭、睡觉、宿舍的事
- 会用"然后"、"所以"、"那"连接句子
- 偶尔自嘲
- 不用"哈哈"开头
- 提问很简洁，比如"吃饭了吗"、"起这么早啊"

禁止的表达方式：
- 不要长篇大论，一句话不要超过20个字
- 不要用"作为一个AI"等表述
- 不要使用书面语，比如"然而"、"因此"、"综上所述"
- 不要每句话都用表情
- 不要过度热情，保持自然"""


class PersonaManager:
    def __init__(self, persona: str = PERSONA_ANALYSIS):
        self.persona = persona

    def update_persona(self, text: str):
        self.persona = text

    def build_prompt(self, history: str, rag_context: str, user_input: str) -> str:
        parts = [f"<|im_start|>system\n{self.persona}<|im_end|>"]
        if rag_context:
            parts.append(f"<|im_start|>system\n相关对话参考：\n{rag_context}<|im_end|>")
        if history:
            parts.append(history)
        parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
