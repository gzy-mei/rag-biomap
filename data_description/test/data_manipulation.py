# data_manipulation.py

import pandas as pd
import os

def extract_name_columns_from_excel(excel_path, output_csv):
    # 存储最终的结构化数据
    data_list = []

    # 加载整个Excel文件
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    for sheet in sheet_names:
        try:
            df = xls.parse(sheet)

            # 严格查找列名为“名称”的列
            if "名称" not in df.columns:
                print(f"[跳过] {sheet} 中未找到列名完全为“名称”的列")
                continue

            col_data = df["名称"].dropna().astype(str)
            for value in col_data:
                data_list.append({
                    "sheet名称": sheet,
                    "原列名": "名称",
                    "内容": value.strip()
                })

        except Exception as e:
            print(f"[错误] 无法处理 Sheet：{sheet}，原因：{e}")
            continue

    # 转为DataFrame保存
    result_df = pd.DataFrame(data_list)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到 {output_csv}，共提取 {len(result_df)} 条记录。")

if __name__ == "__main__":
    excel_path = "dataset/VTE-PTE-CTEPH研究数据库.xlsx"
    output_csv = "data_description/test/标准术语合并结果.csv"
    extract_name_columns_from_excel(excel_path, output_csv)
