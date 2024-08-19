import os

import dotenv

import tqdm
import json

from torch import multiprocessing

from qwen2_api import api_generation


# 加载环境变量
def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        # FINE_TUNE_DATA_OUTPUT_FILE=../data_pool/fineturning_data_pool/fineturning_data.jsonl
        # LABEL_DATA_FILE=../data_pool/label_pool/labels.jsonl
        # REFERENCE_DATA_FILE=../data_pool/test_pool/biological_data_slices_labeled.jsonl
        # INSTRUCTION_DATA_FILE=../data_pool/test_pool/instruction_data_labeled.jsonl
        config_ = {
            "output_file": os.getenv("FINE_TUNE_DATA_OUTPUT_FILE"),
            "request_batch_size": int(os.getenv("FINE_TUNE_DATA_BATCH_SIZE")),
            "combination_file": os.getenv("FINE_SOURCE_DATA_FILE")
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 从文件中加载对应的组合数据
def get_combinations(combination_file_, start_index_):
    with open(combination_file_, 'r', encoding='utf-8') as temp_f:
        combinations_ = []
        for line_ in temp_f:
            combination = json.loads(line_)
            if int(combination["id"]) >= start_index_:
                combinations_.append(combination)

    return combinations_


if __name__ == "__main__":
    # 使用spawn启动子进程，确保cuda可以多进程执行
    multiprocessing.set_start_method("spawn")

    # 获取配置
    config = get_config()
    request_batch_size = config["request_batch_size"]
    combination_file = config["combination_file"]
    output_file = config["output_file"]

    # 确定开始位置
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        start_index = 0
    else:
        start_index = json.loads(lines[-1])["id"] + 1

    # 调用api生成数据
    combinations = get_combinations(combination_file, start_index)
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(start_index, len(combinations), request_batch_size)):
            batch_prompts = [combination["prompt"] for combination in combinations[i:i + request_batch_size]]
            results = api_generation(batch_prompts)
            for j in range(len(batch_prompts)):
                result = results[j]
                response = result.get("response")
                index = int(combinations[i + j]["id"])
                record = {
                    "id": index,
                    "reference_slice": combinations[index]["reference_slice"],
                    "input": combinations[index]["instruction"],
                    "output": response,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(f"已写入{index}条数据\n")
