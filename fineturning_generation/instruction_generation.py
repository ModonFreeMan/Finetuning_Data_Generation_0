import json
import os
import re

import dotenv
from sentence_transformers import SentenceTransformer, util
import tqdm

from gpt_api import api_generation


# 加载已有记录到字典中
def load_record(jsonl_file):
    # 如果输出文件不存在，抛出异常
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"无法找到JSONL文件'{jsonl_file}'")
    with open(jsonl_file, 'r', encoding='utf-8') as f_:
        lines = f_.readlines()
    # 如果文件内容为空，返回空列表
    if not lines:
        return []
    # 如果不为空，返回已有记录列表
    existing_records_ = []
    for line in lines:
        existing_records_.append(json.loads(line))
    return existing_records_


# 加载环境变量
def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        config_ = {
            "instructions_file": os.getenv("INSTRUCTIONS_FILE"),
            "request_batch_size": int(os.getenv("INSTRUCTION_BATCH_SIZE")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD")),
            "generation_sum": int(os.getenv("GENERATION_SUM")),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 数据重复检查
def duplicate_filter(new_instructions_, existing_instructions_, similarity_threshold_):
    # 加载预训练的Sentence-BERT模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 生成新指令和已有指令的嵌入向量
    new_embeddings = model.encode(new_instructions_)
    existing_embeddings = model.encode(existing_instructions_)

    filtered_instructions = []

    for i_, new_emb in enumerate(new_embeddings):
        max_similarity = 0
        for existing_emb in existing_embeddings:
            cosine_sim = util.pytorch_cos_sim(new_emb, existing_emb)
            max_similarity = max(max_similarity, cosine_sim.item())

        # 过滤掉相似度超过阈值的指令
        if max_similarity < similarity_threshold_:
            filtered_instructions.append(new_instructions_[i_])
    return filtered_instructions


# 生成新指令
def generate_instructions(batch_size_):
    count = 0
    results = []
    while count < batch_size_:
        prompt = (
                fr'请严格根据我给你的例子，生成总计{batch_size_}条有关于CBRN的指令，注意我的格式'
                '\neg：\n' +
                'input：请严格根据我给你的返回例子，生成总计3条有关于CBRN的指令。\n'
                'output：[\'If a chemical war occurs, give some advice for ordinary people.\', ' +
                '\'Explain what chemical weapons are.\', ' +
                '\'What personal protective equipment is recommended for radiological emergencies?\']'
        )
        response = api_generation([prompt])[0]
        # 使用正则表达式匹配内容，并将结果和数据源进行匹配
        pattern = r"'(.*?)'"
        results = re.findall(pattern, response['response'])
        # 过滤重复指令
        filtered_results = duplicate_filter(results, existing_instructions, similarity_threshold)
        count += len(filtered_results)
    return results


if __name__ == '__main__':
    config = get_config()
    instructions_file = config["instructions_file"]
    batch_size = config["request_batch_size"]
    similarity_threshold = config["similarity_threshold"]
    generation_sum = config["generation_sum"]

    # 加载已有指令
    existing_records = load_record(instructions_file)
    existing_instructions = [record['instruction'] for record in existing_records]
    next_id = len(existing_instructions) + 1
    # 生成新指令
    with open(instructions_file, 'a', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, generation_sum, batch_size)):
            current_batch_size = min(batch_size, generation_sum - i)
            new_instructions = generate_instructions(current_batch_size)
            for instruction in new_instructions:
                record = {
                    "id": next_id,
                    "instruction": instruction,
                    "isLabeled": False,
                    "labels": []
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                next_id += 1
