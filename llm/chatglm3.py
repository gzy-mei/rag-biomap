from transformers import AutoTokenizer, AutoModel
import torch
import re

# 本地模型路径
model_dir = "/home/gzy/root/.cache/modelscope/hub/models/ZhipuAI/chatglm3-6b"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model.eval()

def generate_with_chatglm3(prompt: str) -> str:
    try:
        response, _ = model.chat(tokenizer, prompt, history=[])
        # 提取编号（仅支持1~3）
        match = re.search(r"\b([1-3])\b", response)
        if match:
            return match.group(1)
        elif "不合适" in response or "都不" in response or "没有一个" in response:
            return ""
        else:
            return ""
    except Exception as e:
        print(f"❌ chatglm3调用失败：{e}")
        return ""
