import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_MODEL_NAME, BASE_MODEL_PATH, BASE_MODEL_DIR, LORA_DIR,
    GRADIO_PORT, MAX_HISTORY_LENGTH, MAX_NEW_TOKENS,
    TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY,
)
from src.memory import ChatMemory
from src.rag import RAGRetriever
from src.persona import PersonaManager, PERSONA_ANALYSIS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

lora_path = Path(str(LORA_DIR))
if lora_path.exists() and (lora_path / "adapter_config.json").exists():
    print("加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
else:
    print("未找到 LoRA 权重，使用基座模型")
    model = base_model

model.eval()

print("初始化记忆模块...")
memory = ChatMemory()
print("初始化 RAG 检索器...")
# rag = RAGRetriever()
rag = None
persona = PersonaManager(PERSONA_ANALYSIS)


def chat(user_input: str, history: list):
    history = history or []
    conv_id = None

    if not hasattr(chat, "conv_id") or chat.conv_id is None:
        chat.conv_id = memory.create_conversation()

    conv_id = chat.conv_id
    memory.add_message(conv_id, "user", user_input)

    recent = memory.get_conversation(conv_id, MAX_HISTORY_LENGTH)
    history_str = "\n".join(
        f"<|im_start|>{r[0]}\n{r[1]}<|im_end|>" for r in recent
    )

    rag_results = rag.search(user_input) if rag else []
    rag_context = "\n".join(rag_results) if rag_results else ""

    prompt = persona.build_prompt(history_str, rag_context, user_input)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    memory.add_message(conv_id, "assistant", response)
    if rag:
        rag.add_messages([user_input, response], [f"query_{conv_id}", f"resp_{conv_id}"])

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, history

def new_conversation():
    chat.conv_id = memory.create_conversation()
    return []


chat.conv_id = None


with gr.Blocks(title="微信聊天记录智能体", css="footer {visibility: hidden}") as demo:
    gr.Markdown("# 微信聊天记录智能体")
    chatbot = gr.Chatbot(label="对话", height=500)
    msg = gr.Textbox(label="输入消息", placeholder="说点什么...")
    with gr.Row():
        send = gr.Button("发送")
        clear = gr.Button("新对话")
    state = gr.State([])

    send.click(chat, [msg, state], [chatbot, state]).then(lambda: "", outputs=[msg])
    msg.submit(chat, [msg, state], [chatbot, state]).then(lambda: "", outputs=[msg])
    clear.click(new_conversation, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT, share=False)
