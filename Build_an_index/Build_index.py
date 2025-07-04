import pandas as pd
import numpy as np
import requests
import os

def get_embedding(text, model='nomic-embed-text', server_url='http://localhost:11434/api/embeddings'):
    response = requests.post(server_url, json={
        'model': model,
        'prompt': text
    })
    response.raise_for_status()
    return response.json()['embedding']

def vectorize_csv_column(csv_path, save_path_npy, column_index=2):
    df = pd.read_csv(csv_path)
    texts = df.iloc[1:, column_index].dropna().astype(str).tolist()  # 跳过第一行
    embeddings = []

    for i, text in enumerate(texts):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            print(f"[{i+1}/{len(texts)}] ✅ 成功嵌入：{text}")
        except Exception as e:
            print(f"[{i+1}] ❌ 嵌入失败：{text}，原因：{e}")

    np.save(save_path_npy, np.array(embeddings, dtype=np.float32))
    print(f"\n🎉 向量化完成！保存到：{save_path_npy}，共 {len(embeddings)} 条")

if __name__ == "__main__":
    csv_path = "/home/gzy/rag-biomap/data_description/标准术语合并结果.csv"
    save_path_npy = "/home/gzy/rag-biomap/Build_an_index/standard_terms.npy"
    vectorize_csv_column(csv_path, save_path_npy)
