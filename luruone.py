import pandas as pd
import os

# 文件路径配置
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库1.xlsx"

# 加载非标准数据，取前两行病人数据
non_std_df = pd.read_excel(non_standard_path, header=0)
data_preview = non_std_df.head(2)

# 加载匹配结果
match_df = pd.read_excel(match_result_path, sheet_name=0)

# 加载标准知识库中的两个 sheet
sheet_names = ["患者基线信息", "病案首页信息"]
std_excel = pd.read_excel(standard_path, sheet_name=None)

# 遍历每个 sheet，先插入两列“录入1”“录入2”
for sheet_name in sheet_names:
    sheet_df = std_excel[sheet_name].copy()
    if "名称" not in sheet_df.columns:
        continue
    name_index = sheet_df.columns.get_loc("名称")
    sheet_df.insert(name_index + 1, "录入1", "")
    sheet_df.insert(name_index + 2, "录入2", "")
    std_excel[sheet_name] = sheet_df

# 遍历匹配表中的每一行
for _, row in match_df.iterrows():
    raw_header = str(row["原始表头"])
    gt_name = str(row["GT标准答案"])
    is_match = str(row["是否匹配GT"]).strip().lower() == "true"

    # 获取非标准数据中原始表头对应的前两条值
    if raw_header not in data_preview.columns:
        continue
    value1, value2 = data_preview[raw_header].tolist()
    if not is_match:
        value1 = value2 = "NA"

    # 查找标准术语是否存在于任一 sheet 的“名称”列
    for sheet_name in sheet_names:
        df = std_excel[sheet_name]
        match_idx = df[df["名称"].astype(str) == gt_name].index
        if not match_idx.empty:
            std_excel[sheet_name].loc[match_idx[0], "录入1"] = value1
            std_excel[sheet_name].loc[match_idx[0], "录入2"] = value2

# 保存新文件
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for sheet_name in sheet_names:
        std_excel[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ 已完成写入，并保存为：{output_path}")
