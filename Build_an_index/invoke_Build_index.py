import pandas as pd
import numpy as np
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


def build_index_from_csv(csv_path, save_path_npy, column_index=2, verbose=False, failed_log_path=None):
    # 明确告诉pandas第一行为表头，自动跳过第一行数据，从第二行开始读取
    df = pd.read_csv(csv_path, header=0)  # 第一行作为header

    # 取指定列（整数索引）所有数据，转换为字符串列表
    texts = df.iloc[:, column_index].dropna().astype(str).tolist()

    embeddings = []
    failed_texts = []

    for i, text in enumerate(texts):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            if verbose:
                print(f"[{i + 1}/{len(texts)}] ✅ 嵌入成功：{text}")
        except Exception as e:
            print(f"[{i + 1}] ❌ 嵌入失败：{text[:20]}，原因：{e}")
            failed_texts.append({"index": i, "text": text, "error": str(e)})

    np.save(save_path_npy, np.array(embeddings, dtype=np.float32))
    print(f"\n🎉 向量化完成！共成功嵌入 {len(embeddings)} 条，保存到：{save_path_npy}")

    if failed_log_path and failed_texts:
        os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
        pd.DataFrame(failed_texts).to_csv(failed_log_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 嵌入失败项记录已保存到：{failed_log_path}")
