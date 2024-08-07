import os

import dotenv

import tqdm
import json

from gpt_api import api_generation


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
            "label_data_file": os.getenv("LABEL_DATA_FILE"),
            "reference_data_file": os.getenv("REFERENCE_DATA_FILE"),
            "instruction_data_file": os.getenv("INSTRUCTION_DATA_FILE"),
            "request_batch_size": int(os.getenv("FINE_TUNE_DATA_BATCH_SIZE"))
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


# 组装提示词
def get_prompts(labels_, reference_data_, instruction_data_):
    prompts_ = []
    for label in labels_:
        matching_instructions = [inst for inst in instruction_data_ if label in inst['labels']]
        matching_references = [ref for ref in reference_data_ if label in ref['labels']]
        if not matching_instructions:
            continue
        for inst in matching_instructions:
            if not matching_references:
                continue
            for ref in matching_references:
                prompt = (
                        f"请根据我给出的题目:\"{inst['instruction']}\"以及可供参考的数据:\"{ref['slice']}\"，" +
                        "生成微调数据，微调数据必须包含input以及output\n" +
                        "example：\n" +
                        "input: Explain what chemical weapons are.\n" +
                        "output: " +
                        "Chemical weapons are toxic chemicals designed to harm or kill humans, animals, " +
                        "or plants as an act of warfare. These weapons exploit the toxic properties of " +
                        "chemical substances rather than their explosive properties to achieve their objectives."
                )
                prompts_.append(prompt)

    return prompts_


if __name__ == "__main__":
    # 获取配置
    config = get_config()
    request_batch_size = config["request_batch_size"]

    # 根据label池生成分组instruction和reference数据
    # 读取label池数据
    label_data = []
    with open(config["label_data_file"], "r", encoding='utf-8') as f:
        for line in f:
            label_data.append(json.loads(line))
    # 读取reference
    reference_data = []
    with open(config["reference_data_file"], "r", encoding='utf-8') as f:
        for line in f:
            reference_data.append(json.loads(line))
    # 读取instruction
    instruction_data = []
    with open(config["instruction_data_file"], "r") as f:
        for line in f:
            instruction_data.append(json.loads(line))
    # 获取prompts
    labels = [label['label'] for label in label_data]
    prompts = get_prompts(labels, reference_data, instruction_data)
    # 调用api生成
    with open(config["output_file"], "w") as f:
        for i in tqdm.tqdm(range(0, len(prompts), request_batch_size)):
            batch_prompts = prompts[i:i + request_batch_size]
            results = api_generation(batch_prompts)
            for j in range(len(batch_prompts)):
                result = results[j]
                response = result.get("response")
                index = i + j
                # 处理response字段，将其转换为包含input和output的字典
                output_lines = response.split("\n")
                input_value = output_lines[0].replace("input: ", "")
                output_value = output_lines[1].replace("output: ", "")
                if output_value == "" or input_value == "":
                    index -= 1
                    continue
                record = {
                    "id": index,
                    "input": input_value,
                    "output": output_value
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
