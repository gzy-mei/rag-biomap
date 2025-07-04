import numpy as np

# 加载向量文件
embeddings = np.load("/home/gzy/rag-biomap/Build_an_index/standard_terms.npy")

# 检查基本信息
print("向量数组形状:", embeddings.shape)  # 应为 (文本数量, 向量维度)
print("第一条向量示例 (前10维):", embeddings[0][:10])  # 查看部分向量