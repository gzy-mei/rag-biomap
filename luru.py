import pandas as pd
import os

# ===== 路径配置 =====
standard_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
non_standard_path = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
match_result_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/中位数(3).xlsx"
output_path = "/home/gzy/rag-biomap/dataset/Results/VTE-PTE-CTEPH研究数据库1.xlsx"

# ===== 读取数据 =====
standard_xls = pd.ExcelFile(standard_path)
non_standard_df = pd.read_excel(non_standard_path, nrows=2)  # 只取前两行
match_df = pd.read_excel(match_result_path, sheet_name="匹配结果")

# ===== 创建输出Writer =====
output_writer = pd.ExcelWriter(output_path, engine='openpyxl')

# ===== 遍历两个sheet =====
for sheet_name in ["患者基线信息", "病案首页信息"]:
    std_df = standard_xls.parse(sheet_name)

    # === 只做一次：在“名称”右侧插入“录入1”“录入2”列 ===
    if "录入1" not in std_df.columns and "录入2" not in std_df.columns:
        name_idx = std_df.columns.get_loc("名称")
        std_df.insert(name_idx + 1, "录入1", "")
        std_df.insert(name_idx + 2, "录入2", "")

    # === 遍历匹配表格 ===
    for _, row in match_df.iterrows():
        raw_header = row["原始表头"]
        gt_term = str(row["GT标准答案"]).strip()
        match_flag = bool(row["是否匹配GT"])

        # 如果GT为空或nan，跳过
        if gt_term == "" or gt_term.lower() == "nan":
            continue

        # 查找标准知识库中是否有匹配行
        match_rows = std_df["名称"] == gt_term
        if not match_rows.any():
            continue

        if match_flag:
            # 如果非标准表中有对应列，就取前两行
            if raw_header in non_standard_df.columns:
                values = non_standard_df[raw_header].fillna("").astype(str).tolist()
                if len(values) < 2:
                    values += [""] * (2 - len(values))
                std_df.loc[match_rows, "录入1"] = values[0]
                std_df.loc[match_rows, "录入2"] = values[1]
            else:
                std_df.loc[match_rows, "录入1"] = "NA"
                std_df.loc[match_rows, "录入2"] = "NA"
        else:
            std_df.loc[match_rows, "录入1"] = "NA"
            std_df.loc[match_rows, "录入2"] = "NA"

    # 写入对应sheet
    std_df.to_excel(output_writer, sheet_name=sheet_name, index=False)

# ===== 保存文件 =====
output_writer.close()
print("✅ 写入完成，输出路径为：", output_path)
