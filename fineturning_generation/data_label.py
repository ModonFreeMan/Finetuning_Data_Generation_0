import json
import os
import re

import tqdm
from dotenv import load_dotenv
from torch import multiprocessing

from qwen2_api import api_generation


# 加载尚未被标记的数据到未标记数据列表中
def load_unlabeled_data(jsonl_file):
    results = []
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"无法找到待标记数据文件'{jsonl_file}'")
    with open(jsonl_file, 'r', encoding='utf-8') as f_:
        lines_ = f_.readlines()
        # 如果文件内容为空,抛出异常
        if not lines_:
            raise FileNotFoundError(f"JSONL文件'{jsonl_file}'内容为空")
        else:
            for line in lines_:
                record_ = json.loads(line)
                # 将所有未标记的数据加入列表
                results.append(record_)
    return results


# 加载环境变量
def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    # 读取环境变量
    try:
        load_dotenv()
        config_ = {
            "input_file": os.getenv('DATA_LABEL_INPUT_FILE'),
            "output_file": os.getenv('DATA_LABEL_OUTPUT_FILE'),
            "request_batch_size": int(os.getenv('LABEL_BATCH_SIZE')),
            "label_pool": os.getenv('LABEL_POOL_PATH'),
            "label_type": os.getenv('LABEL_TYPE'),
        }
        if config_['label_type'] == '':
            raise ValueError('未指定待标记数据类型')
        return config_
    except ValueError as e_:
        print(f'环境变量配置错误: {e_}')
        exit(1)


# 加载标记池文件
def load_label_pool(label_pool_file):
    labels_ = []
    if not os.path.exists(label_pool_file):
        raise FileNotFoundError(f"无法找到标记池文件'{label_pool_file}'")
    with open(label_pool_file, 'r', encoding='utf-8') as f_:
        lines_ = f_.readlines()
        # 如果文件内容为空,抛出异常
        if not lines_:
            raise FileNotFoundError(f"标记池文件'{label_pool_file}'内容为空")
        else:
            for line in lines_:
                record_ = json.loads(line)
                labels_.append(record_['label'])
    return labels_


if __name__ == '__main__':

    # 使用spawn启动子进程，确保cuda可以多进程执行
    multiprocessing.set_start_method("spawn")

    # 加载环境变量
    config = get_config()
    input_file = config['input_file']
    output_file = config['output_file']
    request_batch_size = config['request_batch_size']
    label_pool = config['label_pool']
    label_type = config['label_type']

    # 加载待标记json数据
    try:
        unlabeled_datas = load_unlabeled_data(input_file)
        unlabeled_datas.sort(key=lambda x: x['id'])
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 加载标记池标记
    try:
        labels = load_label_pool(label_pool)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Constructing prompts
    prompts = []
    for i in range(0, len(unlabeled_datas)):
        data = unlabeled_datas[i][label_type]
        prompt = (
                fr'Please, based on the label set I provide: {labels}, only select labels from this set (multiple selections are allowed, but you must choose at least one), '
                fr'label the following data: [{data}]' +
                '\neg:\n' +
                'input: In the event of a sarin gas attack, the first thing to do is to remain calm and quickly seek shelter in a safe area away from the attack site. '
                'If possible, try to leave the affected area and notify local emergency services. Also, try to avoid breathing air from the attack site, covering your nose and mouth with a wet cloth or mask to reduce the risk of inhaling toxic gases. '
                'While escaping the site, pay close attention to local media and official announcements for the latest safety information and instructions. '
                'When encountering rescue personnel, follow their commands and cooperate with their rescue efforts. '
                'The most important thing is to remain calm and rational, avoiding panic and rash actions to ensure your safety and the safety of others.\n' +
                'output: [\'Emergency Response\', \'Chemical Weapons\']'
        )
        prompts.append(prompt)
    # 使用正则表达式匹配内容，并将结果和数据源进行匹配
    pattern = r"'(.*?)'"
    # 检查输出文件
    if not os.path.exists(output_file):
        print(f"输出文件'{output_file}'不存在")
        exit(1)

    # 先读取文件内容，获取上次写入的位置
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        start_index = 0
    else:
        start_index = json.loads(lines[-1])["id"] + 1
        if start_index >= len(unlabeled_datas):
            print('所有数据已经标记完毕')
            exit(1)

        # 检查输出文件
        if not os.path.exists(output_file):
            print(f"输出文件'{output_file}'不存在")
            exit(1)

        # 先读取文件内容，获取上次写入的位置
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            start_index = 0
        else:
            start_index = json.loads(lines[-1])["id"] + 1
            if start_index >= len(unlabeled_datas):
                print('所有数据已经标记完毕')
                exit(1)

    with open(output_file, 'a', encoding='utf-8') as f:
        if label_type == 'slice':
            for i in tqdm.tqdm(range(start_index, len(unlabeled_datas), request_batch_size)):
                batch_prompts = prompts[i:i + request_batch_size]
                responses = api_generation(batch_prompts)
                for j in range(len(batch_prompts)):
                    response = responses[j]['response']
                    result = re.findall(pattern, response)
                    index = i + j
                    record = {
                        "id": unlabeled_datas[index]['id'],
                        "source": unlabeled_datas[index]['source'],
                        "slice": unlabeled_datas[index].get(f'{label_type}', ''),
                        "offset": unlabeled_datas[index]['offset'],
                        "isLabeled": True,
                        "labels": result
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
        else:
            for i in tqdm.tqdm(range(start_index, len(unlabeled_datas), request_batch_size)):
                batch_prompts = prompts[i:i + request_batch_size]
                responses = api_generation(batch_prompts)
                for j in range(len(batch_prompts)):
                    response = responses[j]['response']
                    result = re.findall(pattern, response)
                    index = i + j
                    record = {
                        "id": unlabeled_datas[index]['id'],
                        "instruction": unlabeled_datas[index].get(f'{label_type}', ''),
                        "isLabeled": True,
                        "labels": result
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(f"已标记{index+1}条数据\n")
