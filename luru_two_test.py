import pandas as pd
import os
import shutil
import re
from collections import defaultdict

# === 路径配置 ===
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库5.xlsx"
log_path = "/home/gzy/rag-biomap/dataset/Results/debug_log_final.txt"

# === 创建结果目录 ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === 清空日志 ===
with open(log_path, "w", encoding="utf-8") as log:
    log.write("✅ 脚本开始运行...\n\n")

def clean_text(s):
    s = str(s)
    s = re.sub(r'\s+', '', s)
    s = s.replace('\u3000', '').replace('\u200b', '')
    return s.strip()

# === 复制原始文件到新路径 ===
shutil.copyfile(standard_path, output_path)

# === 加载数据 ===
non_std_df = pd.read_excel(non_standard_path)
non_std_df.columns = [clean_text(c) for c in non_std_df.columns]

match_df = pd.read_excel(match_result_path)
match_df["原始表头"] = match_df["原始表头"].map(clean_text)
match_df["LLM选择"] = match_df["LLM选择"].map(clean_text)
match_df["是否匹配GT"] = match_df["是否匹配GT"].astype(str).str.lower() == "true"

# 提取前两位病人的数据
# patient1 = non_std_df.iloc[0].astype(str).to_dict()
# patient2 = non_std_df.iloc[1].astype(str).to_dict()
def clean_dict(d):
    return {k: ("" if pd.isna(v) else str(v).strip()) for k, v in d.items()}

patient1 = clean_dict(non_std_df.iloc[0])
patient2 = clean_dict(non_std_df.iloc[1])


# === 读取目标 Excel 的所有 sheet ===
all_sheets = pd.read_excel(output_path, sheet_name=None)
sheets_to_update = ["患者基线信息", "病案首页信息"]

# === 插入“录入1”、“录入2”列 ===
for sheet in sheets_to_update:
    df = all_sheets[sheet]
    name_index = df.columns.get_loc("名称")
    if "录入1" not in df.columns:
        df.insert(name_index + 1, "录入1", "")
        df.insert(name_index + 2, "录入2", "")
    all_sheets[sheet] = df

# === 构建 llm -> 多个原始表头的映射 ===
llm_to_raw = defaultdict(list)
llm_match_gt = {}
for _, row in match_df.iterrows():
    llm = row["LLM选择"]
    raw = row["原始表头"]
    match = row["是否匹配GT"]
    if llm:
        llm_to_raw[llm].append(raw)
        llm_match_gt[llm] = match

# === 遍历目标 sheet 并填入值 ===
for sheet in sheets_to_update:
    df = all_sheets[sheet]
    for i, row in df.iterrows():
        name_field = clean_text(row["名称"])

        if name_field in llm_to_raw:
            raw_headers = llm_to_raw[name_field]
            match_gt = llm_match_gt.get(name_field, False)

            if not match_gt:
                df.at[i, "录入1"] = "FALSE"
                df.at[i, "录入2"] = "FALSE"
                continue

            vals1 = [patient1.get(rh, "").strip() for rh in raw_headers]
            vals2 = [patient2.get(rh, "").strip() for rh in raw_headers]

            def get_consistent_value(vals):
                clean_vals = [v for v in vals if v != ""]
                if not clean_vals:
                    return ""
                return clean_vals[0] if all(v == clean_vals[0] for v in clean_vals) else "N/A"

            df.at[i, "录入1"] = get_consistent_value(vals1)
            df.at[i, "录入2"] = get_consistent_value(vals2)
        else:
            df.at[i, "录入1"] = "NA"
            df.at[i, "录入2"] = "NA"
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"❓ sheet: {sheet} 字段【{name_field}】未被模型选择，赋值NA\n")

    all_sheets[sheet] = df

# === 写入最终结果 ===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for sheet, df in all_sheets.items():
        df.columns = [clean_text(c) for c in df.columns]
        df.to_excel(writer, sheet_name=sheet, index=False, na_rep="")


print(f"✅ 最终文件保存于：{output_path}")
print(f"📄 日志记录保存于：{log_path}")
