import json
import os

import dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import tqdm


def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        config_ = {
            "instruction_data_file": os.getenv("INSTRUCTION_DATA_FILE"),
            "sentence_bert_model": os.getenv("SENTENCE_BERT_MODEL"),
            "dbname": os.getenv("MILVUS_DB_NAME"),
            "dbhost": os.getenv("MILVUS_DB_HOST"),
            "dbport": os.getenv("MILVUS_DB_PORT"),
            "collection_name": os.getenv("MILVUS_COLLECTION_NAME"),
            "batch_size": int(os.getenv("COMBINATION_BATCH_SIZE")),
            "nprobe": int(os.getenv("NPROBE")),
            "limit": int(os.getenv("SEARCH_LIMIT")),
            "reference_data_file": os.getenv("REFERENCE_DATA_FILE"),
            "combination_data_file": os.getenv("COMBINATION_DATA_FILE"),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


def search_embeddings(nprobe_, collection_, instruction_embeddings_, limit_):
    search_params = {
        "metric_type": "L2",  # 可以根据创建索引时的设置调整，如 "IP"
        "params": {"nprobe": nprobe_}
    }

    results_ = collection_.search(
        instruction_embeddings_,
        "embedding",
        search_params,
        limit=limit_
    )

    return results_


if __name__ == '__main__':
    # 加载环境变量
    config = get_config()
    # 从环境变量中获取配置
    dbname = config['dbname']
    dbhost = config['dbhost']
    dbport = config['dbport']
    collection_name = config['collection_name']
    instruction_data_file = config['instruction_data_file']
    reference_data_file = config['reference_data_file']
    combination_data_file = config['combination_data_file']
    sentence_ber_model = config['sentence_bert_model']
    batch_size = config['batch_size']
    nprobe = config['nprobe']
    limit = config['limit']

    # 获取instruction数据
    instructions = []
    with open(instruction_data_file, "r", encoding='utf-8') as f:
        for line in f:
            instructions.append(json.loads(line)['instruction'])
    # 获取slice数据并按照id排序
    slices = []
    with open(reference_data_file, "r", encoding='utf-8') as f:
        for line in f:
            slices.append(json.loads(line))
    slices.sort(key=lambda x: x['id'])

    # 加载sentence-bert模型
    print("加载Sentence-BERT模型...\n")
    model = SentenceTransformer(sentence_ber_model)

    # 连接到 Milvus
    print("连接到 Milvus...\n")
    connection = connections.connect(db_name=dbname, host=dbhost, port=dbport)

    # 查询数据
    # 获取或创建集合
    print("获取集合...\n")
    collection = Collection(name=collection_name)
    print("开始加载集合\n")
    collection.load()
    print("集合加载成功\n")

    # 对每个指令生成嵌入向量，并执行搜索
    with open(combination_data_file, 'a', encoding='utf-8') as f:
        print("开始生成组合...\n")
        for i in tqdm.tqdm(range(0, len(instructions), batch_size)):
            batch_instructions = [instruction for instruction in instructions[i:i + batch_size]]
            batch_combinations = []
            for inst in batch_instructions:
                instruction_embedding = model.encode([inst], prompt_name="query")
                results = search_embeddings(nprobe, collection, instruction_embedding, limit)
                if results is None:
                    continue
                # 生成组合并写入文件
                for result in results:
                    for hit_id in result.ids:  # 遍历每个搜索命中的 id
                        slice_ = slices[hit_id].get('slice')
                        prompt = (
                            f"Please provide a comprehensive explanation of [{inst}]. Use the following information "
                            f"if it is relevant,"
                            f"or provide a well-informed response based on general knowledge. Ensure the explanation "
                            f"is accurate and thorough: [{slice_}]")
                        combination = {
                            "instruction": inst,
                            "slice": slice_,
                            "prompt": prompt
                        }
                        batch_combinations.append(combination)  # 确保组合被添加到批次列表中
            # 将数据写入文件
            for combo in batch_combinations:
                f.write(json.dumps(combo) + '\n')
            f.flush()
    # 关闭连接
    connections.disconnect(alias=collection_name)
