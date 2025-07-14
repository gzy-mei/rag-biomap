import pandas as pd
import os
import shutil
import re

# === 路径配置 ===
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库1.xlsx"
log_path = "/home/gzy/rag-biomap/dataset/Results/debug_log.txt"

# === 创建结果目录 ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === 创建并清空日志文件 ===
with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write("✅ 调试日志开始\n\n")

# === 步骤1：复制原文件为新文件（保留所有 sheet）===
shutil.copyfile(standard_path, output_path)
# print(f"✅ 已复制原文件到：{output_path}")

# === 步骤2：加载非标准数据，获取前两行真实值 ===
non_std_df = pd.read_excel(non_standard_path, header=0)

# 清洗函数
def clean_text(s):
    s = str(s)
    s = re.sub(r'\s+', '', s)
    s = s.replace('\u3000', '')
    s = s.replace('\u200b', '')
    return s.strip()

# 清洗列名
non_std_df.columns = [clean_text(col) for col in non_std_df.columns]
data_preview = non_std_df.head(2)

# === 步骤3：加载匹配结果 ===
match_df = pd.read_excel(match_result_path, sheet_name=0)
match_df["原始表头"] = match_df["原始表头"].map(clean_text)
match_df["LLM选择"] = match_df["LLM选择"].map(clean_text)LLM选择

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

# === 步骤6：遍历匹配结果并写入值，同时记录调试信息 ===
for _, row in match_df.iterrows():
    raw_header = row["原始表头"]
    gt_name = row["LLM选择"]
    is_match = str(row["是否匹配GT"]).strip().lower() == "true"

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"👉 当前匹配表原始表头为：{raw_header}\n")
        log_file.write(f"👉 当前非标准数据列名为：{list(data_preview.columns)}\n")

    if raw_header not in data_preview.columns:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"⚠️ 原始表头 {raw_header} 不在非标准数据列中，跳过\n\n")
        continue

    # value1, value2 = data_preview[raw_header].tolist()
    # if not is_match:
    #     value1 = value2 = "NA"
    #
    # for sheet_name in sheets_to_update:
    #     df = all_sheets[sheet_name]
    #     match_idx = df[df["名称"].astype(str).map(clean_text) == gt_name].index
    #     if not match_idx.empty:
    #         all_sheets[sheet_name].loc[match_idx[0], "录入1"] = value1
    #         all_sheets[sheet_name].loc[match_idx[0], "录入2"] = value2
    # 提取前两个非空值
    matched_values = non_std_df[raw_header].dropna().astype(str).tolist()
    value1 = matched_values[0] if len(matched_values) > 0 else "N/A"
    value2 = matched_values[1] if len(matched_values) > 1 else "N/A"

    # 不匹配 GT 的情况设为 NA
    if not is_match:
        value1 = value2 = "NA"

    # ✅ 写入调试信息：包括是否匹配GT + 前两个值
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"👉 GT 标准答案为：{gt_name}\n")
        log_file.write(f"👉 是否匹配GT：{is_match}\n")
        log_file.write(f"👉 表中该列的前两个值：{value1}, {value2}\n\n")

    # ✅ 写入录入值到指定 sheet 中
    for sheet_name in sheets_to_update:
        df = all_sheets[sheet_name]  # ✅ 明确引用
        match_idx = df[df["名称"].astype(str).map(clean_text) == gt_name].index

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"🧪 GT字段 {gt_name} 在 sheet【{sheet_name}】中匹配到的 index：{match_idx.tolist()}\n")

        if not match_idx.empty:
            df.loc[match_idx[0], "录入1"] = value1
            df.loc[match_idx[0], "录入2"] = value2

            all_sheets[sheet_name] = df  # ✅ 必须写回，否则不会保存！

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"✅ 写入 sheet【{sheet_name}】的【{gt_name}】行，录入值：{value1}, {value2}\n")

            # print(df.loc[match_idx[0]])
        else:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"❌ 未在 sheet【{sheet_name}】中找到 GT字段【{gt_name}】对应行，跳过写入\n")

# ✅ 保存中间调试文件，查看是否真的写入了内容
debug_df = all_sheets["患者基线信息"]
debug_path = "/home/gzy/rag-biomap/dataset/Results/debug_患者基线信息.xlsx"
debug_df.to_excel(debug_path, index=False)
print(f"🐛 写入前调试文件已导出：{debug_path}")


# === 步骤7：写入所有 sheet 到新的 Excel 文件（包括未修改的 sheet）===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
    for sheet_name in all_sheets:
        df = all_sheets[sheet_name]

        # 清洗列名（以防写入错误）
        df.columns = [clean_text(c) for c in df.columns]

        df.to_excel(writer, sheet_name=sheet_name, index=False)


print(f"✅ 修改后的文件已保存到：{output_path}")
print(f"📄 调试日志已保存到：{log_path}")
