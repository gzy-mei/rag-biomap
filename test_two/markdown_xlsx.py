import pandas as pd

# 输入输出路径
input_path = '/home/gzy/rag-biomap/test_two/result.md.ZY050002401765'
output_path = '/home/gzy/rag-biomap/test_two/ZY050002401765.xlsx'

# 读取 Markdown 表格
df = pd.read_table(input_path, sep="|", engine="python", skiprows=2)

# 去除空列和前后空白字符
df = df.dropna(axis=1, how="all")
df.columns = [col.strip() for col in df.columns]
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 保存为 Excel 文件
df.to_excel(output_path, index=False)

print(f"✅ 成功转换为 Excel：{output_path}")
