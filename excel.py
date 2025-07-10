import pandas as pd

# 路径设置
csv_path = "/home/gzy/rag-biomap/导出数据第1~1000条数据_病案首页 -(1).csv"
gt_path = "/home/gzy/rag-biomap/GT1.xlsx"
output_path = "/home/gzy/rag-biomap/GT1_对比结果.xlsx"

# 1. 读取 CSV 表头
csv_df = pd.read_csv(csv_path, encoding='utf-8')
csv_headers = csv_df.columns.tolist()

# 2. 读取 GT 表格
gt_df = pd.read_excel(gt_path)

# 3. 获取 GT 第一列（原始表头）
gt_first_column = gt_df.iloc[:, 0].fillna("").astype(str).tolist()

# 4. 对齐长度
max_len = max(len(gt_first_column), len(csv_headers))
gt_first_column += [""] * (max_len - len(gt_first_column))
csv_headers += [""] * (max_len - len(csv_headers))

# 补足 DataFrame 行数
if len(gt_df) < max_len:
    extra_rows = pd.DataFrame(index=range(len(gt_df), max_len))
    gt_df = pd.concat([gt_df, extra_rows], ignore_index=True)

# ✅ 5. 追加“CSV表头”列
gt_df["CSV表头"] = csv_headers

# ✅ 6. 追加“是否一致”列
gt_df["是否一致"] = ["是" if gt_first_column[i].strip() == csv_headers[i].strip() else "否" for i in range(max_len)]

# 7. 保存
gt_df.to_excel(output_path, index=False)
print(f"✅ 对比完成，结果保存到：{output_path}")
