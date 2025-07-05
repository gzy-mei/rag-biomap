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
    base_text = "出院其他诊断编码1"
    compare_texts = [
        "出重症监护室时间",
        "1.一般医疗服务费",
        "疾病编码",
        "其他诊断疾病编码"
    ]

    all_texts = [base_text] + compare_texts

    # 获取所有文本嵌入
    print("🔍 正在获取文本嵌入...")
    embeddings = [get_embedding(t) for t in all_texts]
    embeddings = np.array(embeddings)

    # 计算相似度（只对 base_text 与其他进行比较）
    base_vec = embeddings[0].reshape(1, -1)
    other_vecs = embeddings[1:]
    sim_scores = cosine_similarity(base_vec, other_vecs)[0]

    # 输出和保存
    print("📊 计算结果：")
    for text, score in zip(compare_texts, sim_scores):
        print(f"'{base_text}' 与 '{text}' 的相似度为：{score:.6f}")

    df = pd.DataFrame({
        "文本1": [base_text] * len(compare_texts),
        "文本2": compare_texts,
        "相似度": [round(score, 6) for score in sim_scores]
    })

    output_path = "test_similarity/cosine_similarity/出院其他诊断编码1_向量相似度对比.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 相似度结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
