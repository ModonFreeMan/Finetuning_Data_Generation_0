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
            "dbhost": os.getenv("MILVUS_DB_HOST"),
            "dbport": os.getenv("MILVUS_DB_PORT"),
            "collection_name": os.getenv("MILVUS_COLLECTION_NAME"),
            "instruction_data_file": os.getenv("REQUEST_GENERATION_INPUT_FILE"),
            "request_data_file": os.getenv("REQUEST_GENERATION_OUTPUT_FILE"),
            "sentence_bert_model": os.getenv("REQUEST_GENERATION_MODEL"),
            "batch_size": int(os.getenv("REQUEST_GENERATION_BATCH_SIZE")),
            "nprobe": int(os.getenv("REQUEST_GENERATION_NPROBE")),
            "limit": int(os.getenv("REQUEST_GENERATION_LIMIT")),
            "device": os.getenv("REQUEST_GENERATION_DEVICE"),
        }
        return config_
    except ValueError as e_:
        print(f"环境变量配置错误: {e_}")
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
    dbhost = config['dbhost']
    dbport = config['dbport']
    collection_name = config['collection_name']
    instruction_data_file = config['instruction_data_file']
    request_data_file = config['request_data_file']
    sentence_ber_model = config['sentence_bert_model']
    batch_size = config['batch_size']
    nprobe = config['nprobe']
    limit = config['limit']
    device = config['device']

    # 获取instruction数据
    instructions = []
    with open(instruction_data_file, "r", encoding='utf-8') as f:
        for line in f:
            instructions.append(json.loads(line)['instruction'])

    # 加载sentence-bert模型
    print("加载Sentence-BERT模型...\n")
    model = SentenceTransformer(sentence_ber_model)

    # 连接到 Milvus
    print("连接到 Milvus...\n")
    connections.connect(alias="default", host=dbhost, port=dbport)

    # 获取集合
    print("获取集合...\n")
    collection = Collection(name=collection_name)
    print("开始加载集合\n")
    collection.load()
    print("集合加载成功\n")

    # 对每个指令生成嵌入向量，并执行搜索
    with open(request_data_file, 'a', encoding='utf-8') as f:
        print("开始生成组合...\n")
        for i in tqdm.tqdm(range(0, len(instructions), batch_size)):
            batch_instructions = [instruction for instruction in instructions[i:i + batch_size]]
            batch_requests = []
            try:
                # 批量生成嵌入
                batch_embeddings = model.encode(
                    sentences=batch_instructions,
                    device=device,
                )
                # 批量执行嵌入搜索
                batch_combinations = []
                for inst, embedding in zip(batch_instructions, batch_embeddings):
                    slices = []
                    # 对单个嵌入执行搜索
                    results = search_embeddings(nprobe, collection, embedding, limit)
                    # 提取搜索结果的slice
                    if results:
                        for hits in results:
                            for hit in hits:
                                slices.append(hit.entity.get("slice"))
                    # 使用提示工程扩展指令
                    instruction = (
                        f"Please provide a comprehensive explanation of [{inst}]."
                        f"Use the information I given in the context if it is relevant,"
                        f"or provide a well-informed response based on general knowledge."
                    )
                    request = {
                        "instruction": instruction,
                        "contexts": slices,
                    }
                    batch_requests.append(request)
                # 将组合结果写入文件
                for req in batch_requests:
                    f.write(json.dumps(req) + '\n')
                f.flush()
            except Exception as e:
                print(f"生成组合或搜索时出错: {e}")
    # 关闭连接
    connections.disconnect(alias="default")
