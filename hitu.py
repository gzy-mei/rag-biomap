import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

# ✅ 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 文件路径设置
excel_path = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison/阈值设置30%.xlsx"
output_dir = "/home/gzy/rag-biomap/dataset/Matching_Results_Comparison"
output_path = os.path.join(output_dir, "相似度分布图.png")

# 读取数据
df = pd.read_excel(excel_path)
print("列名预览：", df.columns.tolist())

if "最高相似度" not in df.columns:
    raise ValueError("找不到列：最高相似度，请检查Excel格式")

scores = df["最高相似度"].dropna().tolist()

# 计算统计量
median_score = np.median(scores)
total_count = len(scores)

# 定义区间并统计
bin_edges = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, float('inf')]
bin_labels = ['0', '(0, 2.5]', '(2.5, 5]', '(5, 7.5]',
             '(7.5, 10]', '(10, 12.5]', '(12.5, 15]',
             '(15, 17.5]', '(17.5, 20]', '>20']

# 使用pd.cut进行分类
binned = pd.cut(scores, bins=bin_edges, labels=bin_labels[1:], right=True)
counts = binned.value_counts().sort_index()

# 单独统计0值
zero_count = sum(score == 0 for score in scores)

# 构建分布DataFrame - 确保长度一致
distribution_data = {
    '分数区间': ['等于0'] + bin_labels[1:],
    '数量': [zero_count] + counts.tolist(),
    '占比(%)': [zero_count/total_count*100] + (counts/total_count*100).tolist()
}

distribution_df = pd.DataFrame(distribution_data)

# 打印统计信息
print("\n=== 相似度分数分布统计 ===")
print(f"中位数: {median_score:.4f}")
print(f"总数据量: {total_count}")
print("\n分数区间分布:")
print(distribution_df.to_string(index=False, float_format="%.2f"))

# 绘制直方图
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(scores, bins=30, color='skyblue', edgecolor='black')

# 添加中位数线
plt.axvline(median_score, color='red', linestyle='--', linewidth=1.5)
plt.text(median_score*1.05, max(n)*0.9, f'中位数 = {median_score:.2f}', color='red')

plt.title(f"Top1 向量相似度分数分布（中位数={median_score:.2f}）")
plt.xlabel("Top1 相似度分数")
plt.ylabel("频数")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(output_path, dpi=300)
print(f"\n✅ 图像已保存至：{output_path}")

plt.show()