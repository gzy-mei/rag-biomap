import pandas as pd

# 路径配置
llm_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison_LLMOnly/LLM直接匹配结果.xlsx"
gt_path = "/home/gzy/rag-biomap/dataset/GT.xlsx"
output_path = "/home/gzy/rag-biomap/threshold_test/only_llm/LLM匹配对比结果_含准确率.xlsx"

# 加载两个表格
llm_df = pd.read_excel(llm_path)
gt_df = pd.read_excel(gt_path)

# 取 GT 第二列作为标准答案（重命名方便）
gt_answers = gt_df.iloc[:, 1].astype(str).str.strip()  # 去除多余空格但保留“空字符串”本身
llm_choices = llm_df["LLM选择"].astype(str).str.strip()

# 保证行数一致
assert len(llm_df) == len(gt_answers), "两个文件行数不一致，无法比较"

# 构建新 DataFrame
result_df = pd.DataFrame({
    "原始表头": llm_df["原始表头"],
    "LLM选择": llm_choices,
    "标准答案": gt_answers,
})

# 判断是否匹配成功
result_df["是否匹配成功"] = result_df["LLM选择"] == result_df["标准答案"]

# 计算准确率
accuracy = result_df["是否匹配成功"].mean()
print(f"✅ 匹配准确率: {accuracy:.2%}")

# 保存结果
result_df.to_excel(output_path, index=False)
print(f"📁 结果保存至: {output_path}")
