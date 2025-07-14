import pandas as pd
import os
import shutil
import re

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

# 标准化列名：去除首尾空格和不可见字符
data_preview.columns = data_preview.columns.map(lambda x: str(x).strip())

# 标准化列名：去除不可见字符、全角空格等


def clean_text(s):
    s = str(s)
    s = re.sub(r'\s+', '', s)  # 删除所有空白字符，包括空格、换行、制表符等
    s = s.replace('\u3000', '')  # 删除全角空格
    s = s.replace('\u200b', '')  # 删除零宽空格
    return s.strip()

# 清洗所有非标准数据的列名
non_std_df.columns = [clean_text(col) for col in non_std_df.columns]
data_preview = non_std_df.head(2)

# 同样清洗匹配表的表头字段
match_df["原始表头"] = match_df["原始表头"].map(clean_text)
match_df["GT标准答案"] = match_df["GT标准答案"].map(clean_text)

# ✅ 遍历匹配结果并写入值
for _, row in match_df.iterrows():
    raw_header = row["原始表头"]
    gt_name = row["GT标准答案"]
    is_match = str(row["是否匹配GT"]).strip().lower() == "true"

    # 显示 debug 信息，确认列名实际情况
    if raw_header not in data_preview.columns:
        print(f"⚠️ 原始表头 {raw_header} 不在非标准数据列中。实际列名为: {list(data_preview.columns)}")
        continue

    value1, value2 = data_preview[raw_header].tolist()
    if not is_match:
        value1 = value2 = "NA"

    for sheet_name in sheets_to_update:
        df = all_sheets[sheet_name]
        match_idx = df[df["名称"].astype(str).map(clean_text) == gt_name].index
        if not match_idx.empty:
            all_sheets[sheet_name].loc[match_idx[0], "录入1"] = value1
            all_sheets[sheet_name].loc[match_idx[0], "录入2"] = value2



# === 步骤7：写入所有 sheet 到新的 Excel 文件（包括未修改的 sheet）===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ 修改后的文件已保存到：{output_path}")
