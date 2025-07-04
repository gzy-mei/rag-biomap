# nomic_embed_text.py

import requests

def get_embedding(text: str, model_name="nomic-embed-text", host="http://localhost:11434"):
    """
    调用本地 Ollama 嵌入模型获取文本向量表示
    :param text: 输入的文本字符串
    :param model_name: 模型名称，默认使用 nomic-embed-text
    :param host: Ollama 服务地址，默认 localhost:11434
    :return: 向量（列表形式）
    """
    url = f"{host}/api/embeddings"
    payload = {
        "model": model_name,
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"调用嵌入模型失败: {e}")
        return None
