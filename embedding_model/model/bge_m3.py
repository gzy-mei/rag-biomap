import requests

def get_embedding(text, model='bge-m3', server_url='http://localhost:11434/api/embeddings'):
    """
    调用本地 Ollama 接口，将文本转为嵌入向量（使用 bge-m3 模型）
    """
    response = requests.post(server_url, json={
        'model': model,
        'prompt': text
    })
    response.raise_for_status()
    return response.json()['embedding']
