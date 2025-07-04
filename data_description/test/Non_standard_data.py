import pandas as pd

input_excel = "/home/gzy/rag-biomap/dataset/导出数据第1~1000条数据_病案首页-.xlsx"
output_csv = "/home/gzy/rag-biomap/data_description/test/header_row.csv"

try:
    # 读取不把第一行当表头，所有内容都当数据
    df = pd.read_excel(input_excel, header=None)

    # 取第一行数据，转成Series（一维数组）
    first_row_series = df.iloc[0, :]

    # 转成DataFrame，一列多行
    first_row_df = first_row_series.to_frame().reset_index(drop=True)

    # 保存为csv，无索引，无表头
    first_row_df.to_csv(output_csv, index=False, header=False, encoding='utf-8-sig')

    print(f"成功提取第一行内容并保存到: {output_csv}")

except FileNotFoundError:
    print(f"错误：找不到输入文件 {input_excel}")
except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")
