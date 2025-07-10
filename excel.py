import pandas as pd

# 路径定义
csv_path = "/home/gzy/rag-biomap/导出数据第1~1000条数据_病案首页 -(1).csv"
gt_path = "/home/gzy/rag-biomap/GT1.xlsx"
output_path = "/home/gzy/rag-biomap/GT1_对比结果.xlsx"

# 1. 读取 CSV 的第一行（表头）
csv_df = pd.read_csv(csv_path, encoding='utf-8')
csv_headers = csv_df.columns.tolist()

# 2. 读取 GT.xlsx 的第一列
gt_df = pd.read_excel(gt_path)
gt_column = gt_df.iloc[:, 0].fillna("").astype(str).tolist()

# 3. 补齐 GT DataFrame 行数
max_len = max(len(gt_column), len(csv_headers))
gt_column += [""] * (max_len - len(gt_column))  # GT列补齐
csv_headers += [""] * (max_len - len(csv_headers))  # 表头补齐

# 如果 gt_df 行数不足，手动扩展行
if len(gt_df) < max_len:
    extra_rows = pd.DataFrame(index=range(len(gt_df), max_len))
    gt_df = pd.concat([gt_df, extra_rows], ignore_index=True)

# 添加列
gt_df["CSV表头"] = csv_headers
gt_df["是否一致"] = ["是" if gt_column[i] == csv_headers[i] else "否" for i in range(max_len)]

# 保存
gt_df.to_excel(output_path, index=False)
print(f"✅ 比对完成，结果保存到：{output_path}")
