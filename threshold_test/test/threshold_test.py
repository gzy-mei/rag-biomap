import pandas as pd

# === 文件路径 ===
match_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/匹配结果对比-bge-m3_BM25_20250708_1009.xlsx"
gt_path = "/home/gzy/rag-biomap/dataset/GT.xlsx"
output_path = "/home/gzy/rag-biomap/threshold_test/test/匹配结果对比-bge-m3_BM25_含GT判断.xlsx"

# === 加载数据 ===
match_df = pd.read_excel(match_path)
gt_df = pd.read_excel(gt_path)

# === 删除“平均相似度”列（如果有）===
if "平均相似度" in match_df.columns:
    match_df = match_df.drop(columns=["平均相似度"])

# === 提取GT标准答案列（第二列）===
gt_answers = gt_df.iloc[:, 1].astype(str).apply(lambda x: x if x == " " else x.strip())
llm_choices = match_df["LLM选择"].astype(str).apply(lambda x: x if x == " " else x.strip())

# === 添加标准答案列与比对结果列 ===
match_df["标准答案"] = gt_answers
match_df["是否匹配GT"] = llm_choices == gt_answers

# === 保存新文件 ===
match_df.to_excel(output_path, index=False)
print(f"✅ 新结果已保存至：{output_path}")

# === 输出准确率 ===
accuracy = match_df["是否匹配GT"].mean()
print(f"🎯 LLM选择与GT的匹配准确率：{accuracy:.2%}")
