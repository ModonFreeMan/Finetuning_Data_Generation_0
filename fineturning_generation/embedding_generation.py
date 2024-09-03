import json
import os
import dotenv
import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility


# 加载环境变量
def get_config():
    try:
        dotenv.load_dotenv()
        config_ = {
            "dbhost": os.getenv("MILVUS_DB_HOST"),
            "dbport": os.getenv("MILVUS_DB_PORT"),
            "collection_name": os.getenv("MILVUS_COLLECTION_NAME"),
            "embedding_dim": int(os.getenv("EMBEDDING_GENERATION_DIM")),
            "slice_max_length": int(os.getenv("EMBEDDING_GENERATION_SLICE_MAX_LENGTH")),
            "nlist": int(os.getenv("EMBEDDING_GENERATION_NLIST")),
            "reference_data_file": os.getenv("EMBEDDING_GENERATION_INPUT_FILE"),
            "sentence_bert_model": os.getenv("EMBEDDING_GENERATION_MODEL"),
            "batch_size": int(os.getenv("EMBEDDING_GENERATION_BATCH_SIZE")),
            "device": os.getenv("EMBEDDING_GENERATION_DEVICE"),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 将数据切片存入数据库
def save_embeddings(slices_, collection_, model_, batch_size_, device_):
    print("将数据切片存入数据库...\n")
    # 分批处理数据
    num_slices = len(slices_)
    for start_idx in tqdm.tqdm(range(0, num_slices, batch_size_)):
        end_idx = min(start_idx + batch_size_, num_slices)
        batch_slices = slices_[start_idx:end_idx]

        # 使用模型生成嵌入向量
        embeddings = model_.encode(
            messages=batch_slices,
            device=device_,
        )

        # 准备插入数据，每项包含嵌入和文本
        data = list(zip(embeddings, batch_slices))

        # 插入数据到 Milvus
        collection_.insert(data)
    print("数据切片存入数据库成功\n")


def create_collection(collection_name_, embedding_dim_, slice_max_length_):
    # 检查集合是否存在
    if utility.has_collection(collection_name_):
        print(f"集合 '{collection_name_}' 已存在，加载现有集合。\n")
        collection_ = Collection(collection_name_)
        collection_.load()
        return collection_

    # 定义集合的 Schema
    print("定义集合的 Schema...\n")
    fields = [
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim_),  # 存储嵌入向量
        FieldSchema(name="slice", dtype=DataType.VARCHAR, max_length=slice_max_length_)   # 存储对应文本
    ]
    schema = CollectionSchema(fields)
    print("集合的 Schema 定义成功\n")

    # 创建集合
    print("创建集合...\n")
    collection_ = Collection(collection_name_, schema)
    print("集合创建成功\n")
    return collection_


def create_index(collection_, nlist_):
    # 创建索引
    print("创建索引...\n")
    collection_.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": nlist_
            }
        }
    )
    print("索引创建成功\n")


if __name__ == '__main__':
    config = get_config()

    # 从环境变量中获取配置
    dbname = config['dbname']
    dbhost = config['dbhost']
    dbport = config['dbport']
    collection_name = config['collection_name']
    embedding_dim = config['embedding_dim']
    reference_data_file = config['reference_data_file']
    sentence_bert_model = config['sentence_bert_model']
    batch_size = config['batch_size']
    slice_max_length = config['slice_max_length']
    nlist = config['nlist']
    device = config['device']

    # 获取slice数据
    slices = []
    with open(reference_data_file, "r", encoding='utf-8') as f:
        for line in f:
            # 过滤掉长度过小的数据
            slice = json.loads(line).get('slice')
            if len(slice) > 100:
                slices.append(slice)

    # 加载Sentence-BERT模型
    print("加载Sentence-BERT模型...\n")
    model = SentenceTransformer(sentence_bert_model)

    # 连接到 Milvus
    print("连接到 Milvus...\n")
    connections.connect(alias="default", host=dbhost, port=dbport)

    try:
        # 创建集合
        collection = create_collection(collection_name, embedding_dim, slice_max_length)
        # 将数据切片存入数据库
        save_embeddings(slices, collection, model, batch_size, device)
        # 创建索引
        create_index(collection, nlist)
    finally:
        # 关闭连接
        connections.disconnect(alias="default")
        print("连接已关闭\n")
