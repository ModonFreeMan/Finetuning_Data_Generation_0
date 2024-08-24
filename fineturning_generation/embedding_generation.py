import json
import os
import dotenv
import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection


# 加载环境变量
def get_config():
    try:
        dotenv.load_dotenv()
        config_ = {
            "reference_data_file": os.getenv("REFERENCE_DATA_FILE"),
            "sentence_bert_model": os.getenv("SENTENCE_BERT_MODEL"),
            "dbname": os.getenv("MILVUS_DB_NAME"),
            "dbhost": os.getenv("MILVUS_DB_HOST"),
            "dbport": os.getenv("MILVUS_DB_PORT"),
            "collection_name": os.getenv("MILVUS_COLLECTION_NAME"),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM")),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE")),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 将数据切片存入数据库
def save_embeddings(slices_, collection_, model_, batch_size_):
    print("将数据切片存入数据库...\n")
    # 分批处理数据
    num_slices = len(slices_)
    for start_idx in tqdm.tqdm(range(0, num_slices, batch_size_)):
        end_idx = min(start_idx + batch_size_, num_slices)
        batch_slices = slices_[start_idx:end_idx]

        # 提取 ID 和文本数据
        ids = [item['id'] for item in batch_slices]
        text_slices = [item['slice'] for item in batch_slices]

        # 生成嵌入向量
        embeddings = model_.encode(text_slices)

        # 准备插入的数据
        data = [
            ids,  # ID 列表
            embeddings.tolist(),  # 嵌入向量列表
        ]

        # 插入数据到 Milvus
        collection_.insert(data)
    print("数据切片存入数据库成功\n")


def create_collection(collection_name_, embedding_dim_):
    # 定义集合的 Schema
    print("定义集合的 Schema...\n")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim_),
    ]
    schema = CollectionSchema(fields, collection_name_)
    print("集合的 Schema 定义成功\n")

    # 创建集合
    print("创建集合...\n")
    collection_ = Collection(collection_name_, schema)
    print("集合创建成功\n")
    return collection_


def create_index(collection_):
    # 创建索引
    print("创建索引...\n")
    collection_.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 1000
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

    # 获取slice数据
    slices = []
    with open(reference_data_file, "r", encoding='utf-8') as f:
        for line in f:
            slices.append(json.loads(line))

    # 加载Sentence-BERT模型
    print("加载Sentence-BERT模型...\n")
    model = SentenceTransformer(sentence_bert_model)

    # 连接到 Milvus
    print("连接到 Milvus...\n")
    connection = connections.connect(db_name=dbname, host=dbhost, port=dbport)

    try:
        # 创建集合
        collection = create_collection(collection_name, embedding_dim)
        # 将数据切片存入数据库
        save_embeddings(slices, collection, model, batch_size)
        # 创建索引
        create_index(collection)
    finally:
        # 关闭连接
        connections.disconnect()
        print("连接已关闭\n")
