import pandas as pd

csv_path = "data_description/test/header_row.csv"

# 读取原始CSV，不过滤空行
df_raw = pd.read_csv(csv_path, header=None)

# dropna 后剩下的行（就是你用于向量化的）
filtered_texts = df_raw[0].dropna().astype(str)

# 查找被过滤掉的行索引
missing_indices = df_raw.index.difference(filtered_texts.index)

# 输出排查信息
print(f"📉 被过滤掉的行数: {len(missing_indices)}")
if len(missing_indices) > 0:
    print("被过滤掉的行内容如下：")
    print(df_raw.loc[missing_indices])
else:
    print("✅ 所有行都被保留，没有行被 dropna 过滤掉。")
