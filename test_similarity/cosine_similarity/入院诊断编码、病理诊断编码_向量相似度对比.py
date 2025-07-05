import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# === 嵌入函数 ===
def get_embedding(text, model='nomic-embed-text', server_url='http://localhost:11434/api/embeddings'):
    response = requests.post(server_url, json={
        'model': model,
        'prompt': text
    })
    response.raise_for_status()
    return response.json()['embedding']


# === 主函数 ===
def main():
    texts = ["入院诊断编码", "病理诊断编码"]

    # 获取嵌入向量
    embeddings = [get_embedding(t) for t in texts]
    embeddings = np.array(embeddings)

    # 计算相似度矩阵
    sim_matrix = cosine_similarity(embeddings)

    # 打印并保存结果
    print(f"'{texts[0]}' 与 '{texts[1]}' 的余弦相似度为：{sim_matrix[0,1]:.6f}")

    # 保存到CSV
    df = pd.DataFrame({
        "文本1": [texts[0]],
        "文本2": [texts[1]],
        "相似度": [round(sim_matrix[0,1], 6)]
    })
    df.to_csv("test_similarity/入院诊断编码、病理诊断编码_向量相似度对比.csv", index=False, encoding='utf-8-sig')
    print("✅ 相似度结果已保存至 test_similarity/入院诊断编码、病理诊断编码_向量相似度对比.csv")


if __name__ == "__main__":
    main()
