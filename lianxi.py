#调用嵌入模型“nomic_embed_text.py”，可以在主函数里写入以下内容

from rag.llm.nomic_embed_text import get_embedding

text = "深静脉血栓形成"
vector = get_embedding(text)

if vector:
    print("嵌入向量维度：", len(vector))
    print("部分内容预览：", vector[:5])
else:
    print("获取嵌入失败")


#在魔塔下载模型的地址
""""下载完成：所有分片文件（pytorch_model-0000X-of-00007.bin）和配置文件均已存在。

总大小：约12GB（符合预期）。

文件完整性：包含关键文件：

模型分片（7个文件）

配置文件（config.json、tokenizer_config.json）

分词器（tokenizer.model）

"""
url =/root/.cache/modelscope/hub/models/ZhipuAI/chatglm3-6b
/home/gzy/root/.cache/modelscope/hub/models/ZhipuAI/chatglm3-6b



#main.py
import pandas as pd
import os
import numpy as np
import pickle
import requests
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------- 配置 ----------------------

# 本地模型设置
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_RERANK_URL = "http://localhost:11434/api/rerank"
EMBED_MODEL = "bge-m3"
RERANK_MODEL = "bge-reranker-v2-m3"

# ChatGLM3 模型路径（transformers加载）
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model.eval()

# ---------------------- 工具函数 ----------------------

def extract_name_column_from_excel(excel_path):
    """
    提取所有sheet中“名称”列内容
    """
    result = []
    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            col = next((c for c in df.columns if "名称" in str(c)), None)
            if col:
                for val in df[col].dropna():
                    result.append((sheet, col, str(val).strip()))
        except Exception:
            continue
    return result

def get_ollama_embedding(text):
    data = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(OLLAMA_EMBED_URL, json=data)
    return response.json()["embedding"]

def cosine_similarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def rerank_with_ollama(query, candidates):
    data = {"model": RERANK_MODEL, "query": query, "responses": candidates}
    response = requests.post(OLLAMA_RERANK_URL, json=data)
    return sorted(response.json()["reranked"], key=lambda x: -x["score"])

def chatglm_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------- 主流程 ----------------------

def main():
    # Step 1: 读取标准检查术语（知识库）
    std_data = extract_name_column_from_excel("VTE-PTE-CTEPH研究数据库.xlsx")
    std_texts = [x[2] for x in std_data]
    std_vectors = [get_ollama_embedding(text) for text in tqdm(std_texts, desc="Embedding知识库")]

    # Step 2: 读取医生病单（用户输入）
    df = pd.read_excel("病单.xlsx")
    user_texts = df["病情描述"].astype(str).tolist()  # 修改为实际列名
    user_results = []

    for text in tqdm(user_texts, desc="处理病单"):
        # Step 3: 向量化 + 相似度计算
        q_vec = get_ollama_embedding(text)
        sims = [cosine_similarity(q_vec, v) for v in std_vectors]
        top_k = sorted(zip(std_texts, sims), key=lambda x: -x[1])[:5]
        candidates = [x[0] for x in top_k]

        # Step 4: 重排
        reranked = rerank_with_ollama(text, candidates)
        final_candidates = [std_texts[x["index"]] for x in reranked]

        # Step 5: 构建prompt + ChatGLM3判断
        prompt = f"""
以下是医生在病单中的原始描述：“{text}”。
请根据以下候选标准检查名称，从中选择最符合医生意图的一个：
候选列表：
""" + "\n".join([f"{i+1}、{c}" for i, c in enumerate(final_candidates)]) + \
        "\n请分析语义，并仅输出候选列表中最符合的标准检查的名称。"

        result = chatglm_generate(prompt)
        user_results.append(result.strip())

    # Step 6: 写入Excel
    df["模型推荐检查项目"] = user_results
    df.to_excel("病单_结果.xlsx", index=False)
    print("处理完成，结果已写入 '病单_结果.xlsx'")

if __name__ == "__main__":
    main()
