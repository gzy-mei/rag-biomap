


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


#run main.py
python -m main

可以使用 PyCharm 快捷键 Ctrl + / 一键注释整块选中区域。
选中整段注释代码后，按Ctrl + /


#主代码中调处理标准数据的
def process_standard_data() -> List[str]:
    #from data_description.invoke_data_manipulaltion_basyxx import extract_name_columns_from_excel
    success = extract_name_columns_from_excel(
        CONFIG["standard_excel"],
        CONFIG["standard_terms_csv"],
        target_sheet="病案首页信息",
        target_column="名称"
    )
    if not success:
        raise RuntimeError("标准术语提取失败")

    df = pd.read_csv(CONFIG["standard_terms_csv"])
    required_columns = ["sheet名称", "内容"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    target_sheet = "病案首页信息"
    df_filtered = df[df["sheet名称"] == target_sheet]
    if df_filtered.empty:
        raise ValueError(f"未找到工作表: {target_sheet}")

    terms = df_filtered["内容"].dropna().astype(str).tolist()

    new_csv_path = "data_description/test/病案首页信息_内容.csv"
    df_filtered[["内容"]].to_csv(new_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存筛选后内容到：{new_csv_path}，共 {len(terms)} 条")

    build_index_from_csv(
        new_csv_path,
        CONFIG["standard_vectors"],
        column_index=0,
        verbose=False
    )
    return terms