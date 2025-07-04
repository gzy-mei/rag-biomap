import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_including_self(csv_path, embedding_path, output_path):
    # 加载文本内容
    df = pd.read_csv(csv_path, header=None)
    texts = df.iloc[:, 0].dropna().astype(str).tolist()

    # 加载向量
    embeddings = np.load(embedding_path)
    if len(texts) != embeddings.shape[0]:
        print(f"❌ 文本数量 {len(texts)} 与向量数量 {embeddings.shape[0]} 不一致")
        return

    # 计算余弦相似度矩阵（包括自己）
    sim_matrix = cosine_similarity(embeddings)

    results = []
    for i, text in enumerate(texts):
        sim_scores = sim_matrix[i]
        max_idx = np.argmax(sim_scores)  # 包含自己时，最大通常是自己

        max_text = texts[max_idx]
        max_score = sim_scores[max_idx]

        results.append({
            "原文本": text,
            "最相似文本": max_text,
            "相似度": round(max_score, 6),
            "是否相同": "是" if text == max_text else "否"
        })

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存相似度比较结果到：{output_path}")

if __name__ == "__main__":
    csv_path = "/home/gzy/rag-biomap/data_description/test/header_row.csv"
    embedding_path = "/home/gzy/rag-biomap/Build_an_index/test/header_terms.npy"
    output_path = "/home/gzy/rag-biomap/test_similarity/cosine_similarity/header_similarity_including_self.csv"

    find_most_similar_including_self(csv_path, embedding_path, output_path)
