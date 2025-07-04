import sys
import os

# 将新的模块路径添加到sys.path，方便import
sys.path.extend([
    "/home/gzy/rag-biomap/data_description",        # 你的data_description目录
    "/home/gzy/rag-biomap/Build_an_index",          # Build_an_index目录
])

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_description.invoke_Non_standard_data import extract_first_row_to_csv
from Build_an_index.invoke_Build_index import get_embedding, build_index_from_csv
from data_description.invoke_data_manipulaltion_basyxx import extract_name_columns_from_excel
from openai import OpenAI
from typing import List, Dict
from Build_an_index.invoke_Non_standard_data_Build_index import vectorize_header_terms
from rank_bm25 import BM25Okapi
import jieba


def debug_data():
    print("==== 调试开始 ====")

    # 1. 查看标准术语CSV文件行数和sheet名称分布
    df_standard = pd.read_csv("/home/gzy/rag-biomap/data_description/test/标准术语_病案首页.csv")
    print(f"标准术语CSV总行数: {len(df_standard)}")
    print("标准术语CSV中各sheet名称计数:")
    print(df_standard["sheet名称"].value_counts())

    # 2. 查看非标准数据CSV行数
    df_header = pd.read_csv("/home/gzy/rag-biomap/data_description/test/header_row.csv", header=None)
    print(f"非标准数据CSV总行数: {len(df_header)}")

    # 3. 查看向量文件内容数量
    try:
        header_vectors = np.load("/home/gzy/rag-biomap/Build_an_index/test/header_terms.npy")
        print(f"header_vectors数量: {header_vectors.shape[0]}")
    except Exception as e:
        print(f"读取header_vectors时出错: {e}")

    try:
        standard_vectors = np.load("/home/gzy/rag-biomap/Build_an_index/test/standard_terms.npy")
        print(f"standard_vectors数量: {standard_vectors.shape[0]}")
    except Exception as e:
        print(f"读取standard_vectors时出错: {e}")

    # 4. 计算相似度时打印异常LLM选择
    header_texts = df_header[0].tolist()
    standard_texts = df_standard["内容"].dropna().astype(str).tolist()

    for h_text in header_texts:
        # 这里简单打印，方便观察，真实调试中可放到相似度计算函数里
        if not h_text or h_text.strip() == "":
            print(f"异常表头文本为空: '{h_text}'")

    print("==== 调试结束 ====")

# 调用这个调试函数，放在main函数的开始或你想检查的位置
if __name__ == "__main__":
    debug_data()



# 初始化OpenAI客户端
client = OpenAI(
    base_url="http://172.16.55.171:7010/v1",
    api_key="sk-cairi"
)

# 配置参数
CONFIG = {
    "non_standard_excel": "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx",
    "standard_excel": "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx",
    "header_csv": "/home/gzy/rag-biomap/data_description/test/header_row.csv",
    "standard_terms_csv": "/home/gzy/rag-biomap/data_description/test/标准术语_病案首页.csv",
    "header_vectors": "/home/gzy/rag-biomap/Build_an_index/test/header_terms.npy",
    "standard_vectors": "/home/gzy/rag-biomap/Build_an_index/test/standard_terms.npy",
    "output_excel": "/home/gzy/rag-biomap/dataset/匹配结果对比.xlsx",
}

# 后续函数保持不变，直接用你之前写的即可

def initialize_directories():
    for path in [CONFIG["header_csv"], CONFIG["header_vectors"]]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def process_non_standard_data() -> List[str]:
    # 先调用你的 extract_first_row_to_csv，确保 CSV 生成成功
    if not extract_first_row_to_csv(CONFIG["non_standard_excel"], CONFIG["header_csv"]):
        raise RuntimeError("非标准数据处理失败")

    # 调用封装好的向量化函数，替代原来的 build_index_from_csv 调用
    vectorize_header_terms(
        CONFIG["header_csv"],
        CONFIG["header_vectors"],
        failed_log_path="/home/gzy/rag-biomap/Build_an_index/test/header_terms_failed.csv"
    )

    # 返回所有文本列表
    return pd.read_csv(CONFIG["header_csv"], header=None)[0].tolist()

def process_standard_data() -> List[str]:
    success = extract_name_columns_from_excel(
        CONFIG["standard_excel"],
        CONFIG["standard_terms_csv"],
        target_sheet="病案首页信息",
        target_column="名称"
    )
    if not success:
        raise RuntimeError("标准术语提取失败")

    df = pd.read_csv(CONFIG["standard_terms_csv"])
    required_columns = ["sheet名称", "内容"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    target_sheet = "病案首页信息"
    df_filtered = df[df["sheet名称"] == target_sheet]
    if df_filtered.empty:
        raise ValueError(f"未找到工作表: {target_sheet}")

    terms = df_filtered["内容"].dropna().astype(str).tolist()

    new_csv_path = "/home/gzy/rag-biomap/data_description/test/病案首页信息_内容.csv"
    df_filtered[["内容"]].to_csv(new_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存筛选后内容到：{new_csv_path}，共 {len(terms)} 条")

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


"""余弦
def calculate_similarities() -> List[Dict]:
   header_vectors = np.load(CONFIG["header_vectors"])
    standard_vectors = np.load(CONFIG["standard_vectors"])
    header_texts = pd.read_csv(CONFIG["header_csv"], header=None)[0].tolist()
    standard_texts = pd.read_csv(CONFIG["standard_terms_csv"])["内容"].tolist()

    results = []
    for h_text, h_vec in zip(header_texts, header_vectors):
        sim_scores = cosine_similarity([h_vec], standard_vectors)[0]
        top_3_indices = np.argsort(sim_scores)[-3:][::-1]
        top_3 = [standard_texts[i] for i in top_3_indices]
        top_scores = [sim_scores[i] for i in top_3_indices]

        """
#bm25
def calculate_similarities() -> List[Dict]:
    header_texts = pd.read_csv(CONFIG["header_csv"], header=None)[0].dropna().astype(str).tolist()
    standard_texts = pd.read_csv(CONFIG["standard_terms_csv"])["内容"].dropna().astype(str).tolist()

    tokenized_corpus = [list(jieba.cut(text)) for text in standard_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    for h_text in header_texts:
        query = list(jieba.cut(h_text))
        scores = bm25.get_scores(query)
        top_3_indices = np.argsort(scores)[-3:][::-1]
        top_3 = [standard_texts[i] for i in top_3_indices]
        top_scores = [scores[i] for i in top_3_indices]

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
    df = pd.DataFrame(results)
    df['匹配成功'] = df.apply(lambda x: x['LLM选择'] in x['候选术语'][0], axis=1)
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
