import pandas as pd
import os

def extract_name_columns_from_multiple_sheets(
    excel_path: str,
    output_csv: str,
    target_sheets: list,
    target_column: str = "名称"
) -> bool:
    """
    从指定Excel文件的多个sheet中提取指定列（如“名称”列），并保存到一个CSV文件中。
    """
    data_list = []

    try:
        for sheet_name in target_sheets:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            if target_column not in df.columns:
                print(f"⚠️ 警告：在 sheet '{sheet_name}' 中未找到列 '{target_column}'，跳过该 sheet。")
                continue

            col_data = df[target_column].dropna().astype(str)
            for value in col_data:
                data_list.append({
                    "sheet名称": sheet_name,
                    "原列名": target_column,
                    "内容": value.strip()
                })

    except Exception as e:
        print(f"❌ 错误：读取 Excel 出错，原因：{e}")
        return False

    # 保存结果
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df = pd.DataFrame(data_list)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到 {output_csv}，共提取 {len(result_df)} 条记录。")
    return True

# 示例调用
if __name__ == "__main__":
    excel_path = "dataset/VTE-PTE-CTEPH研究数据库.xlsx"
    output_csv = "data_description/test/标准术语_前2sheet.csv"
    target_sheets = ["患者基线信息", "病案首页信息"]  # 顺序提取
    extract_name_columns_from_multiple_sheets(excel_path, output_csv, target_sheets)
