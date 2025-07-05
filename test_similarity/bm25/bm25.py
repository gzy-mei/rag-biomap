import pandas as pd
from rank_bm25 import BM25Okapi
import jieba  # 中文分词
import os

def find_most_similar_bm25(csv_path, output_path):
    # 加载文本内容
    df = pd.read_csv(csv_path, header=None)
    texts = df.iloc[:, 0].dropna().astype(str).tolist()

    # 中文分词（BM25 依赖词项匹配）
    tokenized_corpus = [list(jieba.cut(text)) for text in texts]

    # 构建 BM25 模型
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    for i, (text, query_tokens) in enumerate(zip(texts, tokenized_corpus)):
        scores = bm25.get_scores(query_tokens)
        max_idx = scores.argmax()
        max_text = texts[max_idx]
        max_score = round(float(scores[max_idx]), 6)

        results.append({
            "原文本": text,
            "最相似文本": max_text,
            "相似度": max_score,
            "是否相同": "是" if text == max_text else "否"
        })

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存 BM25 相似度结果到：{output_path}")

if __name__ == "__main__":
    csv_path = "data_description/test/header_row.csv"
    output_path = "test_similarity/bm25/header_similarity_bm25.csv"

    find_most_similar_bm25(csv_path, output_path)
