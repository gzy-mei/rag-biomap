import sys
import os
import re
import argparse
import pandas as pd
import numpy as np
import jieba
from openai import OpenAI
from typing import List, Dict
#进行数据集处理
from data_description.invoke_data_manipulaltion_basyxx import extract_name_columns_from_excel
from data_description.invoke_Non_standard_data import extract_first_row_to_csv
#进行向量化
from Build_an_index.invoke_Build_index import get_embedding, build_index_from_csv
from Build_an_index.invoke_Non_standard_data_Build_index import vectorize_header_terms
#计算向量相似度
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi



def debug_data():
    print("==== 调试开始 ====")

    # 1. 查看标准术语CSV文件行数和sheet名称分布
    df_standard = pd.read_csv("data_description/test/标准术语_病案首页.csv")
    print(f"标准术语CSV总行数: {len(df_standard)}")
    print("标准术语CSV中各sheet名称计数:")
    print(df_standard["sheet名称"].value_counts())

    # 2. 查看非标准数据CSV行数
    df_header = pd.read_csv("data_description/test/header_row.csv", header=None)
    print(f"非标准数据CSV总行数: {len(df_header)}")

    # 3. 查看向量文件内容数量
    try:
        header_vectors = np.load("Build_an_index/test/header_terms.npy")
        print(f"header_vectors数量: {header_vectors.shape[0]}")
    except Exception as e:
        print(f"读取header_vectors时出错: {e}")

    try:
        standard_vectors = np.load("Build_an_index/test/standard_terms.npy")
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

# 调用这个调试函数
if __name__ == "__main__":
    debug_data()



# 初始化OpenAI客户端
client = OpenAI(
    base_url="http://172.16.55.171:7010/v1",
    api_key="sk-cairi"
)

# 配置参数
CONFIG = {
    "llm_model": "CAIRI-LLM-reasoner",
    "non_standard_excel": "dataset/导出数据第1~1000条数据_病案首页-.xlsx",
    "standard_excel": "dataset/VTE-PTE-CTEPH研究数据库.xlsx",
    "header_csv": "data_description/test/header_row.csv",
    "standard_terms_csv": "data_description/test/标准术语_病案首页.csv",
    "header_vectors": "Build_an_index/test/header_terms.npy",
    "standard_vectors": "Build_an_index/test/standard_terms.npy",
    "output_excel": "dataset/匹配结果对比.xlsx",
    #"embedding_model": "nomic-embed-text",  # 修改为当前使用的嵌入模型名：bge-m3、nomic-embed-text、mxbai-embed-large还需要再调用函数中修改！！！
    #"similarity_method": "BM25",  # 相似度方法，有：BM25，Cosine
    "output_dir": "dataset/Matching_Results_Comparison"

}


#初始化目录结构： 根据配置文件CONFIG中的路径，自动创建这些路径所在的文件夹目录（如果目录不存在滴话）
def initialize_directories():
    for path in [CONFIG["header_csv"], CONFIG["header_vectors"]]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

#处理非标准数据（表头信息）
def process_non_standard_data() -> List[str]:
    #from data_description.invoke_Non_standard_data import extract_first_row_to_csv
    # 先调用extract_first_row_to_csv，确保CSV生成成功
    #结果生成：data_description/test/header_row.csv
    if not extract_first_row_to_csv(CONFIG["non_standard_excel"], CONFIG["header_csv"]):
        raise RuntimeError("非标准数据处理失败")

    # 调用封装好的向量化函数
    #from Build_an_index.invoke_Non_standard_data_Build_index import vectorize_header_terms
    #结果生成：Build_an_index/test/header_terms.npy
    vectorize_header_terms(
        CONFIG["header_csv"],
        CONFIG["header_vectors"],
        failed_log_path="Build_an_index/test/header_terms_failed.csv"
    )

    # 返回所有文本列表
    return pd.read_csv(CONFIG["header_csv"], header=None)[0].tolist()

#处理标准术语数据（知识库）
def process_standard_data() -> List[str]:
    """
    处理标准术语数据：提取术语列 → 保存CSV → 向量化 → 返回术语列表
    """
    # 直接提取标准术语sheet名称、原列名、内容
    # from data_description.invoke_data_manipulaltion_basyxx import extract_name_columns_from_excel
    #结果生成：data_description/test/标准术语_病案首页.csv
    success = extract_name_columns_from_excel(
        CONFIG["standard_excel"],
        CONFIG["standard_terms_csv"],
        target_sheet="病案首页信息",
        target_column="名称"
    )
    if not success:
        raise RuntimeError("标准术语提取失败")

    # 加载CSV并提取术语（即“内容”列-不会包含“内容”这两个字。）
    df = pd.read_csv(CONFIG["standard_terms_csv"])
    if "内容" not in df.columns:
        raise ValueError("标准术语CSV缺少 '内容' 列")

    terms = df["内容"].dropna().astype(str).tolist()
    print(f"✅ 成功加载标准术语，共 {len(terms)} 条")

    # 向量化“内容”列
    #结果生成：Build_an_index/test/standard_terms.npy
    #from Build_an_index.invoke_Build_index import get_embedding, build_index_from_csv
    build_index_from_csv(
        CONFIG["standard_terms_csv"],     # 直接使用原CSV
        CONFIG["standard_vectors"],       # 保存向量路径
        column_index=2,                   # “内容”列在CSV中的位置
        verbose=False
    )

    return terms

# def generate_with_llm(prompt: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=CONFIG["llm_model"],
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=100
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"⚠️ LLM调用失败：{e}")
#         return "[默认回复]"


# def generate_with_llm(prompt: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="CAIRI-LLM-reasoner",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=100
#         )
#
#         message_obj = response.choices[0].message
#         # 优先读取 content，如果没有就读取 reasoning_content
#         if message_obj.content:
#             return message_obj.content.strip()
#         elif hasattr(message_obj, "reasoning_content") and message_obj.reasoning_content:
#             return message_obj.reasoning_content.strip()
#         else:
#             return "[空响应]"
#     except Exception as e:
#         print(f"⚠️ LLM调用失败：{e}")
#         return "[默认回复]"



# def generate_with_llm(prompt: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=CONFIG["llm_model"],
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=100
#         )
#
#         message_obj = response.choices[0].message
#         raw_content = message_obj.content.strip() if message_obj.content else ""
#         return raw_content
#     except Exception as e:
#         print(f"⚠️ LLM调用失败：{e}")
#         return "[调用失败]"

# def generate_with_llm(prompt: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=CONFIG["llm_model"],
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=100
#         )
#
#         # 打印完整响应结构！
#         print("📦 LLM完整响应结构：")
#         print(response)
#
#         message_obj = response.choices[0].message
#         raw_content = message_obj.content.strip() if message_obj.content else ""
#         return raw_content
#     except Exception as e:
#         print(f"⚠️ LLM调用失败：{e}")
#         return "[调用失败]"


def generate_with_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )

        message_obj = response.choices[0].message

        # 提取 LLM 返回内容（兼容 CAIRI 的 reasoning_content 字段）
        raw_content = None
        if hasattr(message_obj, "content") and message_obj.content:
            raw_content = message_obj.content.strip()
        elif hasattr(message_obj, "reasoning_content") and message_obj.reasoning_content:
            raw_content = message_obj.reasoning_content.strip()
        else:
            raw_content = ""

        print(f"📨 表头：{prompt.split('原始表头：')[-1].splitlines()[0]}")
        print(f"🎯 LLM原始返回：'{raw_content}'")
        print(f"📦 LLM完整响应结构：\n{response}")
        print("📝 ========= LLM 调用记录 =========")

        # 提取编号 1~4
        match = re.search(r'\b([1-4])\b', raw_content)
        if match:
            llm_choice = match.group(1)
        else:
            llm_choice = "N/A"

        return llm_choice

    except Exception as e:
        print(f"⚠️ LLM调用失败：{e}")
        return "N/A"


def detect_similarity_method(func):
    def wrapper(*args, **kwargs):
        method_name = func.__name__.lower()
        if "bm25" in method_name:
            CONFIG["similarity_method"] = "BM25"
        elif "cosine" in method_name:
            CONFIG["similarity_method"] = "Cosine"
        else:
            CONFIG["similarity_method"] = "Unknown"
        return func(*args, **kwargs)
    return wrapper

# =========================
# ⚙️ 阈值配置（相似度阈值）
# =========================
threshold_ratio = 0.35  # 表示相似度最高得分的35%


#bm25
@detect_similarity_method
def calculate_similarities_bm25() -> List[Dict]:
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

        # # 判断是否低于阈值
        # if top_scores[0] < max(scores) * threshold_ratio:
        #     llm_choice_result = ""  # 不调用LLM，直接空字符串
        # else:
        #     prompt = f"""请根据病历表头选择最匹配的标准术语（如果没有合适的请选4）：
        #     原始表头：{h_text}
        #     候选术语：
        #     {chr(10).join(f'{i + 1}. {text}' for i, text in enumerate(top_3))}
        #     4. 无一个候选项匹配
        #
        #     请只返回选择的编号(1-4)，不要解释。"""
        #
        #     llm_choice = generate_with_llm(prompt)
        #     if llm_choice.isdigit() and 1 <= int(llm_choice) <= 3:
        #         llm_choice_result = top_3[int(llm_choice) - 1]
        #     elif llm_choice.strip() == "4":
        #         llm_choice_result = ""
        #     else:
        #         llm_choice_result = "N/A"

        # 判断是否低于阈值
        if top_scores[0] < max(scores) * threshold_ratio:
            llm_choice_result = ""  # 不调用LLM，直接空字符串
        else:
            prompt = f"""请根据病历表头选择最匹配的标准术语（如果没有合适的请选4）：
        原始表头：{h_text}
        候选术语：
        {chr(10).join(f'{i + 1}. {text}' for i, text in enumerate(top_3))}
        4. 无一个候选项匹配

        请只返回选择的编号(1-4)，不要解释。"""

            llm_choice = generate_with_llm(prompt)

            # 🐞 调试输出
            print("📝 ========= LLM 调用记录 =========")
            print(f"📨 表头：{h_text}")
            print(f"🎯 LLM原始返回：'{llm_choice}'")

            # 正则提取 1-4 的编号
            match = re.search(r"\b([1-4])\b", llm_choice)
            if match:
                choice_num = int(match.group(1))
                if choice_num == 4:
                    llm_choice_result = ""
                else:
                    llm_choice_result = top_3[choice_num - 1]
            else:
                llm_choice_result = "N/A"

        results.append({
            "原始表头": h_text,
            "候选术语": top_3,
            "LLM选择": llm_choice_result,
            "最高相似度": top_scores[0],
            "平均相似度": np.mean(top_scores)
        })

    return results


#结果保存函数
def save_results(results: List[Dict]):
    df = pd.DataFrame(results)

    # 删除不需要的列
    df.drop(columns=[col for col in ["平均相似度", "匹配成功"] if col in df.columns], inplace=True)

    # 加载 GT 标准答案（跳过表头）
    gt_path = "/home/gzy/rag-biomap/dataset/GT.xlsx"
    gt_df = pd.read_excel(gt_path, header=0)

    if gt_df.shape[1] < 2:
        raise ValueError("GT.xlsx 必须至少包含两列，第二列为标准答案")

    gt_answers = gt_df.iloc[:, 1].fillna("").astype(str).tolist()
    df["GT标准答案"] = pd.Series(gt_answers[:len(df)])

    # 匹配判断
    df["是否匹配GT"] = df.apply(lambda row: row["LLM选择"] == row["GT标准答案"], axis=1)

    # 统计信息
    total_accuracy = df["是否匹配GT"].mean()
    gt_empty_count = sum(df["GT标准答案"] == "")
    llm_empty = df["LLM选择"] == ""
    gt_empty = df["GT标准答案"] == ""
    llm_not_empty = df["LLM选择"] != ""

    llm_empty_and_gt_empty = df[llm_empty & gt_empty].shape[0]
    llm_empty_total = llm_empty.sum()
    llm_not_empty_total = llm_not_empty.sum()
    llm_not_empty_gt_empty = df[llm_not_empty & gt_empty].shape[0]
    llm_empty_gt_not_empty = df[llm_empty & ~gt_empty].shape[0]

    # 删除“是否匹配GT”列，结果中不保留
    df.drop(columns=["是否匹配GT"], inplace=True)

    # 添加统计信息到 DataFrame 的尾部
    stats = pd.DataFrame([
        ["llm选择与GT标准答案匹配准确率", total_accuracy],
        ["GT标准答案中空值个数", gt_empty_count],
        ["llm选择为空，GT也为空的匹配成功数量", llm_empty_and_gt_empty],
        ["llm选择为空的数量", llm_empty_total],
        ["llm选择非空的数量", llm_not_empty_total],
        ["llm选择非空，但GT是空的数量", llm_not_empty_gt_empty],
        ["llm选择为空，GT不为空的数量", llm_empty_gt_not_empty]
    ], columns=df.columns[:2])  # 用前两列对齐表头

    df_final = pd.concat([df, stats], ignore_index=True)

    # 保存路径
    output_dir = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison"
    os.makedirs(output_dir, exist_ok=True)
    filename = "阈值设置35.xlsx"
    output_path = os.path.join(output_dir, filename)

    # 保存 Excel
    df_final.to_excel(output_path, index=False, engine="openpyxl")

    # 控制台输出（辅助确认）
    print(f"✅ 结果已保存到 {output_path}，共 {len(df)} 条记录 + 统计信息")
    print(f"📊 匹配准确率：{total_accuracy:.6f}")
    print(f"📊 GT为空值：{gt_empty_count}，llm选择为空数量：{llm_empty_total}")
    print(f"📊 llm选择为空 && GT为空（匹配）：{llm_empty_and_gt_empty}")
    print(f"📊 llm选择非空 && GT为空：{llm_not_empty_gt_empty}")
    print(f"📊 llm选择为空 && GT非空：{llm_empty_gt_not_empty}")





def main():
    from Build_an_index.invoke_Build_index import get_embedding


    #自动识别当前使用的嵌入模型--方法：获取get_embedding函数的来源模块路径（比如 bge_m3.py）
    module_path = get_embedding.__module__

    print(f"📁 检测到 get_embedding 来自模块：{module_path}")  # ✅ ：调试输出

    if "bge_m3" in module_path:
        CONFIG["embedding_model"] = "bge-m3"
    elif "nomic_embed_text" in module_path:
        CONFIG["embedding_model"] = "nomic-embed-text"
    elif "mxbai_embed_large" in module_path:
        CONFIG["embedding_model"] = "mxbai-embed-large"
    else:
        CONFIG["embedding_model"] = "unknown"

    #CONFIG["embedding_model"] = get_embedding.__defaults__[0]

    # 自动检测当前启用的相似度计算函数是哪一个-通过全局变量表globals()来检测哪个函数存在，是我有没有“定义”它。
    if "calculate_similarities_bm25" in globals():
        calculate_similarities = calculate_similarities_bm25
    elif "calculate_similarities_cosine" in globals():
        calculate_similarities = calculate_similarities_cosine
    else:
        raise ValueError(
            "未检测到相似度计算函数，请确保保留 calculate_similarities_bm25 或 calculate_similarities_cosine 中的一个")

    initialize_directories()
    print("🔄 处理非标准数据...")
    header_texts = process_non_standard_data()

    print("🔄 处理标准知识库...")
    standard_texts = process_standard_data()

    print("🔍 计算相似度并执行简化RAG流程...")
    results = calculate_similarities()

#运行结果显示：📐 当前使用的相似度计算方法为：......
    print(f"📐 当前使用的相似度计算方法为：{CONFIG['similarity_method']}")

    print("💾 保存结果...")
    save_results(results)

if __name__ == "__main__":
    main()
