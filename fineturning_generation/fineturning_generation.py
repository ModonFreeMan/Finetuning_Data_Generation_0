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
        config_ = {
            "request_file": os.getenv("FINE_TUNE_GENERATION_INPUT_FILE"),
            "output_file": os.getenv("FINE_TUNE_GENERATION_OUTPUT_FILE"),
            "request_batch_size": int(os.getenv("FINE_TUNE_GENERATION_BATCH_SIZE")),
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")
        exit(1)


if __name__ == "__main__":
    # 使用spawn启动子进程，确保cuda可以多进程执行
    multiprocessing.set_start_method("spawn")

    # 获取配置
    config = get_config()
    request_batch_size = config["request_batch_size"]
    request_file = config["request_file"]
    output_file = config["output_file"]

    # 获取请求数据
    with open(request_file, 'r', encoding='utf-8') as temp_f:
        requests = []
        for line_ in temp_f:
            request = json.loads(line_)
            requests.append(request)
    print(f"共读取到{len(requests)}条请求数据\n")

    # 先读取文件内容，获取上次写入的位置
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        start_index = 0
    else:
        start_index = json.loads(lines[-1])["id"] + 1
    print(f"将从第{start_index}行位置开始写入\n")

    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(start_index, len(requests), request_batch_size)):
            batch_requests = [request for request in requests[i:i + request_batch_size]]
            results = api_generation(batch_requests)
            for j in range(len(results)):
                result = results[j]
                response = result.get("response")
                index = i + j
                record = {
                    "id": i + j,
                    "instruction": requests[index].get("instruction"),
                    "input": requests[index].get("contexts"),
                    "output": response,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(f"已写入{index + 1}条数据\n")
