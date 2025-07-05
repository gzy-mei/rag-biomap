import pandas as pd
import numpy as np
import requests
import os

#from embedding_model.model.nomic_embed_text import get_embedding
from embedding_model.model.mxbai_embed_large import get_embedding
#from embedding_model.model.bge_m3 import get_embedding

"""def get_embedding(text, model='nomic-embed-text', server_url='http://localhost:11434/api/embeddings'):
    response = requests.post(server_url, json={
        'model': model,
        'prompt': text
    })
    response.raise_for_status()
    return response.json()['embedding']"""

def vectorize_header_terms(csv_path, save_path_npy, failed_log_path=None):
    df = pd.read_csv(csv_path, header=None)
    texts = df.iloc[:, 0].astype(str).tolist()  # 不再 dropna，保留所有文本

    embeddings = []
    failed_texts = []

    for i, text in enumerate(texts):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            #print(f"[{i+1}/{len(texts)}] ✅ 嵌入成功：{text}")
        except Exception as e:
            print(f"[{i+1}] ❌ 嵌入失败：{text}，原因：{e}")
            failed_texts.append({"index": i, "text": text, "error": str(e)})

    np.save(save_path_npy, np.array(embeddings, dtype=np.float32))
    print(f"\n🎉 嵌入完成，共成功嵌入 {len(embeddings)} 条，保存到：{save_path_npy}")

    if failed_log_path and failed_texts:
        failed_df = pd.DataFrame(failed_texts)
        os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
        failed_df.to_csv(failed_log_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 失败文本已保存到：{failed_log_path}")

if __name__ == "__main__":
    csv_path = "data_description/test/header_row.csv"
    save_path_npy = "Build_an_index/test/header_terms.npy"
    failed_log_path = "Build_an_index/test/header_terms_failed.csv"

    vectorize_header_terms(csv_path, save_path_npy, failed_log_path)
