import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from data_description.invoke_Non_standard_data import extract_first_row_to_csv
from data_description.invoke_data_manipulaltion import extract_name_columns_from_excel
from Build_an_index.invoke_Build_index import get_embedding, build_index_from_csv
import os
from typing import List, Dict
from openai import OpenAI

client = OpenAI(
    base_url="http://172.16.55.171:7010/v1",
    api_key="sk-cairi"
)


# 配置参数
CONFIG = {
    # 文件路径
    "non_standard_excel": "/home/gzy/rag-biomap/导出数据第1~1000条数据_病案首页-.xlsx",
    "standard_excel": "/home/gzy/rag-biomap/VTE-PTE-CTEPH研究数据库.xlsx",
    "header_csv": "/home/gzy/rag-biomap/data_description/header_row.csv",
    "standard_terms_csv": "/home/gzy/rag-biomap/data_description/标准术语合并结果.csv",
    "header_vectors": "/home/gzy/rag-biomap/Build_an_index/header_terms.npy",
    "standard_vectors": "/home/gzy/rag-biomap/Build_an_index/standard_terms.npy",
    "output_excel": "/home/gzy/rag-biomap/匹配结果对比.xlsx",

}


def initialize_directories():
    """确保所有目录存在"""
    for path in [CONFIG["header_csv"], CONFIG["header_vectors"]]:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def process_non_standard_data() -> List[str]:
    """处理非标准数据并返回表头文本列表"""
    if not extract_first_row_to_csv(CONFIG["non_standard_excel"], CONFIG["header_csv"]):
        raise RuntimeError("非标准数据处理失败")
    build_index_from_csv(CONFIG["header_csv"], CONFIG["header_vectors"], column_index=0, verbose=False)
    return pd.read_csv(CONFIG["header_csv"], header=None)[0].tolist()


def process_standard_data() -> List[str]:
    """处理标准知识库数据并返回术语列表"""
    extract_name_columns_from_excel()  # 你已有的函数，生成标准术语合并结果.csv

    df = pd.read_csv(CONFIG["standard_terms_csv"])

    # 确认必要列
    required_columns = ["sheet名称", "内容"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    # 筛选sheet名称为“病案首页信息”的行
    target_sheet = "病案首页信息"
    df_filtered = df[df["sheet名称"] == target_sheet]
    if df_filtered.empty:
        raise ValueError(f"未找到工作表: {target_sheet}")

    # 取“内容”列，去除空值，转字符串列表
    terms = df_filtered["内容"].dropna().astype(str).tolist()

    # 保存新CSV文件到指定路径
    new_csv_path = "/home/gzy/rag-biomap/data_description/病案首页信息_内容.csv"
    df_filtered[["内容"]].to_csv(new_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存筛选后内容到：{new_csv_path}，共 {len(terms)} 条")

    # 生成向量文件
    build_index_from_csv(
        new_csv_path,
        CONFIG["standard_vectors"],
        column_index=0,
        verbose=False
    )
    return terms




def generate_with_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="CAIRI-LLM",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM调用失败：{e}")
        return "[默认回复]"


def calculate_similarities() -> List[Dict]:
    # 加载数据
    header_vectors = np.load(CONFIG["header_vectors"])
    standard_vectors = np.load(CONFIG["standard_vectors"])
    header_texts = pd.read_csv(CONFIG["header_csv"], header=None)[0].tolist()
    standard_texts = pd.read_csv(CONFIG["standard_terms_csv"])["内容"].tolist()

    results = []
    for h_text, h_vec in zip(header_texts, header_vectors):
        # 1. 计算余弦相似度
        sim_scores = cosine_similarity([h_vec], standard_vectors)[0]
        top_3_indices = np.argsort(sim_scores)[-3:][::-1]  # 直接取相似度最高的3个
        top_3 = [standard_texts[i] for i in top_3_indices]
        top_scores = [sim_scores[i] for i in top_3_indices]

        # 2. LLM生成最终选择
        prompt = f"""请根据病历表头选择最匹配的标准术语：

        原始表头：{h_text}
        候选术语：
        {chr(10).join(f'{i + 1}. {text}' for i, text in enumerate(top_3))}

        只需返回选择的编号(1-3)，不要解释。"""

        llm_choice = generate_with_llm(prompt)
        results.append({
            "原始表头": h_text,
            "候选术语": top_3,
            "LLM选择": top_3[int(llm_choice) - 1] if llm_choice.isdigit() else "N/A",
            "最高相似度": top_scores[0],
            "平均相似度": np.mean(top_scores)
        })
    return results


def save_results(results: List[Dict]):
    """保存带有多维度评估的结果"""
    df = pd.DataFrame(results)
    # 添加匹配成功标记
    df['匹配成功'] = df.apply(
        lambda x: x['LLM选择'] in x['候选术语'][0],
        axis=1
    )
    df.to_excel(CONFIG["output_excel"], index=False, engine='openpyxl')
    print(f"✅ 结果已保存到 {CONFIG['output_excel']}，共 {len(df)} 条记录")


def main():
    initialize_directories()
    print("🔄 处理非标准数据...")
    header_texts = process_non_standard_data()

    print("🔄 处理标准知识库...")
    standard_texts = process_standard_data()

    print("🔍 计算相似度并执行简化RAG流程...")
    results = calculate_similarities()

    print("💾 保存结果...")
    save_results(results)


if __name__ == "__main__":
    main()