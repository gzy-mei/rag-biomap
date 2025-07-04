import pandas as pd
import os

def extract_name_columns_from_excel(
    excel_path: str,
    output_csv: str,
    target_sheet: str = "病案首页信息",
    target_column: str = "名称"
) -> bool:
    """
    从指定Excel文件的指定sheet和列提取数据，保存为CSV。
    """
    data_list = []

    try:
        df = pd.read_excel(excel_path, sheet_name=target_sheet)

        if target_column not in df.columns:
            print(f"错误：在'{target_sheet}'表中未找到列 '{target_column}'")
            return False

        col_data = df[target_column].dropna().astype(str)
        for value in col_data:
            data_list.append({
                "sheet名称": target_sheet,
                "原列名": target_column,
                "内容": value.strip()
            })

    except Exception as e:
        print(f"错误：读取 Excel 时出错，原因：{e}")
        return False

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df = pd.DataFrame(data_list)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到 {output_csv}，共提取 {len(result_df)} 条记录。")
    return True


if __name__ == "__main__":
    excel_path = "/home/gzy/rag-biomap/dataset/VTE-PTE-CTEPH研究数据库.xlsx"
    output_csv = "/home/gzy/rag-biomap/data_description/test/标准术语_病案首页.csv"
    extract_name_columns_from_excel(excel_path, output_csv)
