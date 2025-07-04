import pandas as pd


def extract_first_row_to_csv(input_excel_path, output_csv_path):
    """
    从Excel文件中提取第一行数据并保存为CSV文件

    参数:
        input_excel_path (str): 输入的Excel文件路径
        output_csv_path (str): 输出的CSV文件路径

    返回:
        bool: 操作是否成功
    """
    try:
        # 读取Excel文件，不将第一行作为表头
        df = pd.read_excel(input_excel_path, header=None)

        # 提取第一行数据
        first_row_series = df.iloc[0, :]

        # 转换为DataFrame（一列多行）
        first_row_df = first_row_series.to_frame().reset_index(drop=True)

        # 保存为CSV文件
        first_row_df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8-sig')

        print(f"成功提取第一行内容并保存到: {output_csv_path}")
        return True

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_excel_path}")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


def main():
    # 定义文件路径
    input_excel = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
    output_csv = "/home/gzy/rag-biomap/data_description/test/header_row.csv"

    # 调用函数处理文件
    success = extract_first_row_to_csv(input_excel, output_csv)

    if success:
        print("文件处理完成！")
    else:
        print("文件处理失败，请检查错误信息。")


if __name__ == "__main__":
    main()