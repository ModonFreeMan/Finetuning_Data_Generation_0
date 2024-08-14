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

    # 进行提示词组装
    prompts = []
    for i in range(0, len(unlabeled_datas)):
        data = unlabeled_datas[i][label_type]
        prompt = (
                fr'请根据我给你的标记词库：{labels}，只允许从中挑选标记词（可以有多个，必须至少选择一个），'
                fr'对以下数据进行标记：[{data}]' +
                '\neg：\n' +
                'input：如果发生沙林武器袭击，首先要保持冷静并尽快躲避到安全地带，远离袭击现场。'
                '如果可能的话，应该尽快脱离受害区域，并通知当地应急部门。同时，要尽量避免呼吸袭击现场的空气，用湿布或口罩遮住口鼻以减少吸入有毒气体的风险。'
                '在逃离现场的过程中，要密切关注当地媒体和官方通告，以获取最新的安全信息和指示。在遇到救援人员时，要听从他们的指挥和安排，协助他们进行救援工作。'
                '最重要的是，保持冷静和理智，不要恐慌和乱行动，以确保自己和他人的安全。\n' +
                'output：[\'Emergency Response\', \'Chemical Weapons\']'
        )
        prompts.append(prompt)
    # 使用正则表达式匹配内容，并将结果和数据源进行匹配
    pattern = r"'(.*?)'"
    # 检查输出文件
    if not os.path.exists(output_file):
        print(f"输出文件'{output_file}'不存在")
        exit(1)

    with open(output_file, 'a', encoding='utf-8') as f:
        # 获取上次断点位置
        lines = f.readlines()
        # 如果文件内容为空
        if not lines:
            start_index = 0
        else:
            start_index = json.loads(lines[-1])["id"] + 1
            if start_index >= len(unlabeled_datas):
                print('所有数据已经标记完毕')
                exit(1)
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
