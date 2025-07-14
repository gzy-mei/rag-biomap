import pandas as pd
import os
import shutil

# === 路径配置 ===
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库1.xlsx"

# === 步骤1：复制原文件为新文件（保留所有 sheet）===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
shutil.copyfile(standard_path, output_path)
print(f"✅ 已复制原文件到：{output_path}")

# === 步骤2：加载非标准数据，获取前两行真实值 ===
non_std_df = pd.read_excel(non_standard_path, header=0)
data_preview = non_std_df.head(2)

# === 步骤3：加载匹配结果 ===
match_df = pd.read_excel(match_result_path, sheet_name=0)

# === 步骤4：读取新复制文件的所有 sheet（将会进行修改）===
sheets_to_update = ["患者基线信息", "病案首页信息"]
all_sheets = pd.read_excel(output_path, sheet_name=None)

# === 步骤5：在两个目标 sheet 中插入“录入1”“录入2”列 ===
for sheet_name in sheets_to_update:
    df = all_sheets[sheet_name].copy()
    if "名称" not in df.columns:
        continue
    name_index = df.columns.get_loc("名称")
    if "录入1" not in df.columns and "录入2" not in df.columns:
        df.insert(name_index + 1, "录入1", "")
        df.insert(name_index + 2, "录入2", "")
    all_sheets[sheet_name] = df

# === 步骤6：处理每一行匹配结果并写入数据 ===
for _, row in match_df.iterrows():
    raw_header = str(row["原始表头"])
    gt_name = str(row["GT标准答案"])
    is_match = str(row["是否匹配GT"]).strip().lower() == "true"

    # 如果列不存在，跳过
    if raw_header not in data_preview.columns:
        continue

    value1, value2 = data_preview[raw_header].tolist()
    if not is_match:
        value1 = value2 = "NA"

    # 在两个 sheet 中查找目标标准术语并写入值
    for sheet_name in sheets_to_update:
        df = all_sheets[sheet_name]
        match_idx = df[df["名称"].astype(str) == gt_name].index
        if not match_idx.empty:
            all_sheets[sheet_name].loc[match_idx[0], "录入1"] = value1
            all_sheets[sheet_name].loc[match_idx[0], "录入2"] = value2

# === 步骤7：写入所有 sheet 到新的 Excel 文件（包括未修改的 sheet）===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ 修改后的文件已保存到：{output_path}")
