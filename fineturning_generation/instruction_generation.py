import json
import os
import random

import dotenv
from sentence_transformers import SentenceTransformer, util
import tqdm
from torch import multiprocessing

from qwen2_api import api_generation


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
            "generation_sum": int(os.getenv("INSTRUCTION_GENERATION_SUM")),
            "sentence_bert_model": os.getenv("SENTENCE_BERT_MODEL")
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 数据重复检查
def duplicate_filter(new_instructions_, existing_instructions_, similarity_threshold_):
    # 检查新生成的向量是否为空
    if not new_instructions_:
        return []
    # 检查已有的向量是否为空
    if not existing_instructions_:
        return new_instructions_

    # 加载预训练的Sentence-BERT模型
    print("加载Sentence-BERT模型...\n")
    model = SentenceTransformer(config["sentence_bert_model"])

    # 生成新指令和已有指令的嵌入向量
    new_embeddings = model.encode(new_instructions_)
    existing_embeddings = model.encode(existing_instructions_)

    filtered_instructions = []
    print(f"过滤前指令数量：{len(new_instructions_)}")
    for i_, new_emb in enumerate(new_embeddings):
        max_similarity = 0
        for existing_emb in existing_embeddings:
            cosine_sim = util.pytorch_cos_sim(new_emb, existing_emb)
            max_similarity = max(max_similarity, cosine_sim.item())

        # 过滤掉相似度超过阈值的指令
        if max_similarity < similarity_threshold_:
            filtered_instructions.append(new_instructions_[i_])
    print(f"过滤后指令数量：{len(filtered_instructions)}\n")
    return filtered_instructions


# 生成新指令
def generate_instructions(batch_size_, existing_instructions_):
    results = []
    prompts = []
    # 如果已有指令列表为空，使用默认的指令
    for _ in range(batch_size_):
        if len(existing_instructions_) == 0:
            random_instruction = "What can ordinary people do to protect themselves from chemical threats?"
        else:
            # 从已有指令列表中随机选择一条指令构建指令集合
            random_instruction = random.choice(existing_instructions_)
        prompt = (
                fr'Please follow my example, generate one instruction about Chemical or Biological or' +
                'Radiological or Nuclear, just question, do not contain answer' +
                '\neg:\n' +
                fr'{random_instruction}'
        )
        prompts.append(prompt)
    responses = api_generation(prompts)
    for response in responses:
        results.append(response['response'])
    return results


if __name__ == '__main__':
    # 使用spawn启动子进程，确保cuda可以多进程执行
    multiprocessing.set_start_method("spawn")

    config = get_config()
    instructions_file = config["instructions_file"]
    batch_size = config["request_batch_size"]
    similarity_threshold = config["similarity_threshold"]
    generation_sum = config["generation_sum"]

    # 生成新指令
    with open(instructions_file, 'a', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, generation_sum, batch_size)):
            # 加载已有指令嵌入向量
            existing_records = load_record(instructions_file)
            next_id = len(existing_records) + 1
            existing_instructions = [record['instruction'] for record in existing_records]

            # 生成新指令
            current_batch_size = min(batch_size, generation_sum - i)
            new_instructions = generate_instructions(current_batch_size, existing_instructions)

            # 去重
            new_instructions = duplicate_filter(new_instructions, existing_instructions, similarity_threshold)

            # 写入文件
            for instruction in new_instructions:
                record = {
                    "id": next_id,
                    "instruction": instruction,
                    "isLabeled": False,
                    "labels": []
                }
                next_id += 1
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"已生成{next_id}条指令\n")
            # 强制将缓冲区内容写入磁盘
            f.flush()
