# invoke_Build_index.py

import pandas as pd
import numpy as np
import requests

def get_embedding(text, model='nomic-embed-text', server_url='http://localhost:11434/api/embeddings'):
    response = requests.post(server_url, json={
        'model': model,
        'prompt': text
    })
    response.raise_for_status()
    return response.json()['embedding']

def build_index_from_csv(csv_path, save_path_npy, column_index=2, verbose=False):
    df = pd.read_csv(csv_path)
    texts = df.iloc[1:, column_index].dropna().astype(str).tolist()
    embeddings = []

    for i, text in enumerate(texts):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            if verbose and (i+1) % 50 == 0:  # 每50条打印一次进度
                print(f"[{i+1}/{len(texts)}] 处理中...")
        except Exception as e:
            print(f"[{i+1}] ❌ 嵌入失败：{text[:20]}...，原因：{str(e)[:50]}...")

    np.save(save_path_npy, np.array(embeddings, dtype=np.float32))
    if verbose:
        print(f"\n🎉 向量化完成！共 {len(embeddings)} 条")
