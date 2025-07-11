import pandas as pd

# 原本是 CSV，但扩展名错误
csv_path = "dataset/导出数据第1~1000条数据_病案首页-.xlsx"  # 实际是 CSV
# 目标 Excel 文件路径
excel_path = "dataset/导出数据第1~1000条数据_病案首页_fix.xlsx"

# 用 pd.read_csv 正确读取
df = pd.read_csv(csv_path, encoding='utf-8')  # 如出错可改为 encoding='gbk'
# 转为真正的 Excel 格式
df.to_excel(excel_path, index=False)

print(f"✅ 已转换为真实 Excel 文件：{excel_path}")
