import pandas as pd
import os
import shutil
import re

# === 路径配置 ===
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库2.xlsx"

# === 创建输出目录 ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === 清洗函数 ===
def clean_text(s):
    s = str(s)
    s = re.sub(r'\s+', '', s)
    s = s.replace('\u3000', '').replace('\u200b', '')
    return s.strip()

# === 步骤1：复制原文件为新文件（保留所有 sheet） ===
shutil.copyfile(standard_path, output_path)

# === 步骤2：加载非标准数据 ===
non_std_df = pd.read_excel(non_standard_path, header=0)
non_std_df.columns = [clean_text(col) for col in non_std_df.columns]

# === 步骤3：加载匹配结果 ===
match_df = pd.read_excel(match_result_path)
match_df["原始表头"] = match_df["原始表头"].map(clean_text)
match_df["GT标准答案"] = match_df["GT标准答案"].map(clean_text)

# === 步骤4：加载标准知识库所有 sheet ===
sheets_to_update = ["患者基线信息", "病案首页信息"]
all_sheets = pd.read_excel(output_path, sheet_name=None)

# === 步骤5：插入录入1、录入2列 ===
for sheet_name in sheets_to_update:
    df = all_sheets[sheet_name].copy()
    if "名称" not in df.columns:
        continue
    name_idx = df.columns.get_loc("名称")
    if "录入1" not in df.columns:
        df.insert(name_idx + 1, "录入1", "")
    if "录入2" not in df.columns:
        df.insert(name_idx + 2, "录入2", "")
    all_sheets[sheet_name] = df

# === 步骤6：处理匹配表并填入录入值 ===
for _, row in match_df.iterrows():
    raw_header = row["原始表头"]
    gt_name = row["GT标准答案"]
    is_match = str(row["是否匹配GT"]).strip().lower() == "true"

    # 提取前两个真实值（或填充NA）
    if raw_header in non_std_df.columns:
        values = non_std_df[raw_header].dropna().astype(str).tolist()
        value1 = values[0] if len(values) > 0 else "N/A"
        value2 = values[1] if len(values) > 1 else "N/A"
    else:
        value1 = value2 = "N/A"

    if not is_match:
        value1 = value2 = "NA"

    # 写入目标 sheet 中的对应字段行
    for sheet_name in sheets_to_update:
        df = all_sheets[sheet_name]
        match_idx = df[df["名称"].astype(str).map(clean_text) == gt_name].index
        if not match_idx.empty:
            df.loc[match_idx[0], "录入1"] = value1
            df.loc[match_idx[0], "录入2"] = value2
            all_sheets[sheet_name] = df

# === 步骤7：写入所有 sheet 到新的文件 ===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for sheet_name, df in all_sheets.items():
        df.columns = [clean_text(col) for col in df.columns]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ 数据已成功写入：{output_path}")
