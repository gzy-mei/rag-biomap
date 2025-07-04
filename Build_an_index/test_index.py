import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载向量文件和原始文本
embeddings = np.load("/home/gzy/rag-biomap/Build_an_index/standard_terms.npy")
df = pd.read_csv("/home/gzy/rag-biomap/data_description/标准术语合并结果.csv")
texts = df.iloc[1:, 2].dropna().astype(str).tolist()  # 假设第3列是文本

# 2. 检查数据是否对齐
assert len(texts) == len(embeddings), "文本数量与向量数量不匹配！"

# 3. 查询相似文本
query_idx = 0  # 以第一条文本为例
query_vector = embeddings[query_idx].reshape(1, -1)

# 计算余弦相似度
similarities = cosine_similarity(query_vector, embeddings)[0]
top_k = np.argsort(similarities)[-5:][::-1]  # 取相似度最高的5条

# 4. 打印结果
print(f"查询文本: '{texts[query_idx]}'")
print("最相似文本及相似度:")
for idx in top_k:
    print(f"- {texts[idx]} (相似度: {similarities[idx]:.4f})")