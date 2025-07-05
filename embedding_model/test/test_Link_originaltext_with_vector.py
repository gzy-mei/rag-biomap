import pandas as pd
import numpy as np

# 1. 加载原始文本
df = pd.read_csv("data_description/标准术语合并结果.csv")
texts = df.iloc[1:, 2].dropna().astype(str).tolist()  # 假设第3列是文本

# 2. 加载向量文件
embeddings = np.load("Build_an_index/standard_terms.npy")

# 3. 打印前5条文本及其向量
for i in range(5):
    print(f"文本 {i+1}: {texts[i]}")
    print(f"对应向量 (前5维): {embeddings[i][:5]}...\n")