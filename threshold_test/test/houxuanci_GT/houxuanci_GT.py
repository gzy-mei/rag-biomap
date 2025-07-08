import pandas as pd
import os
import ast  # 用于解析候选术语列的字符串列表

# 路径配置
result_path = "/home/gzy/rag-biomap/threshold_test/test/bge_m3_BM25_GT/匹配结果对比-bge-m3_BM25_含GT判断.xlsx"
gt_path = "/home/gzy/rag-biomap/dataset/GT.xlsx"
output_dir = "/home/gzy/rag-biomap/threshold_test/test/houxuanci_GT"
output_file = os.path.join(output_dir, "候选术语第一项与GT比较结果.xlsx")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取原始结果文件
df = pd.read_excel(result_path)

# 读取GT文件第二列
gt_df = pd.read_excel(gt_path, header=0)
gt_answers = gt_df.iloc[:, 1].fillna("").astype(str).tolist()

# 提取“候选术语”第一项
def extract_first_candidate(cell):
    try:
        parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else ""
    except Exception as e:
        print(f"解析失败：{cell}，错误：{e}")
        return ""

df["第一个候选词"] = df["候选术语"].apply(extract_first_candidate)

# 对齐 GT 内容数量
gt_series = pd.Series(gt_answers[:len(df)], name="GT标准答案_第二列")
df["GT标准答案_第二列"] = gt_series

# 判断是否匹配
df["第一个候选词与GT是否匹配"] = df.apply(
    lambda row: row["第一个候选词"] == row["GT标准答案_第二列"],
    axis=1
)

# 保存结果
df.to_excel(output_file, index=False, engine="openpyxl")
print(f"✅ 已保存文件到：{output_file}")
