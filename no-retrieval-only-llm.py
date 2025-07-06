import os
import pandas as pd
from openai import OpenAI
from typing import List, Dict
from data_description.invoke_Non_standard_data import extract_first_row_to_csv

# 初始化OpenAI客户端
client = OpenAI(
    base_url="http://172.16.55.171:7010/v1",
    api_key="sk-cairi"
)

# 配置参数
CONFIG = {
    "non_standard_excel": "dataset/导出数据第1~1000条数据_病案首页-.xlsx",
    "standard_excel": "dataset/VTE-PTE-CTEPH研究数据库.xlsx",
    "header_csv": "data_description/test/header_row.csv",
    "standard_terms_csv": "data_description/test/标准术语_病案首页.csv",
    "output_dir": "dataset/Matching_Results_Comparison_LLMOnly"
}

# 提取标准术语
def extract_standard_terms() -> List[str]:
    from data_description.invoke_data_manipulaltion_basyxx import extract_name_columns_from_excel

    success = extract_name_columns_from_excel(
        CONFIG["standard_excel"],
        CONFIG["standard_terms_csv"],
        target_sheet="病案首页信息",
        target_column="名称"
    )
    if not success:
        raise RuntimeError("标准术语提取失败")

    df = pd.read_csv(CONFIG["standard_terms_csv"])
    if "内容" not in df.columns:
        raise ValueError("缺少内容列")

    return df["内容"].dropna().astype(str).tolist()

# 提取非标准数据
def extract_non_standard_headers() -> List[str]:
    if not extract_first_row_to_csv(CONFIG["non_standard_excel"], CONFIG["header_csv"]):
        raise RuntimeError("非标准表头提取失败")

    return pd.read_csv(CONFIG["header_csv"], header=None)[0].dropna().astype(str).tolist()

# 调用LLM进行匹配
def match_with_llm(header: str, candidates: List[str]) -> str:
    prompt = f"""请根据病历表头选择最匹配的标准术语：
原始表头：{header}
候选术语：
{chr(10).join(f"{i + 1}. {term}" for i, term in enumerate(candidates))}

只需返回最匹配的编号 (1-{len(candidates)})，不要解释。"""

    try:
        response = client.chat.completions.create(
            model="CAIRI-LLM",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        choice = response.choices[0].message.content.strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        return "N/A"
    except Exception as e:
        print(f"⚠️ LLM调用失败：{e}")
        return "[默认回复]"

# 主函数
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("🔄 加载候选术语...")
    standard_terms = extract_standard_terms()

    print("🔄 加载原始表头...")
    headers = extract_non_standard_headers()

    print("🤖 执行LLM匹配...")
    results = []
    for header in headers:
        llm_result = match_with_llm(header, standard_terms)
        results.append({
            "原始表头": header,
            "LLM选择": llm_result
        })

    df = pd.DataFrame(results)
    output_path = os.path.join(CONFIG["output_dir"], "LLM直接匹配结果.xlsx")
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✅ 匹配结果已保存到：{output_path}")

if __name__ == "__main__":
    main()
