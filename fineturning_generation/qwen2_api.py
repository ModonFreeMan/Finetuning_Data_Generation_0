import concurrent.futures
import warnings
import os
from datetime import datetime

import dotenv
from modelscope import AutoModelForCausalLM, AutoTokenizer


def make_request(config_, prompt_, tokenizer, model, device):
    messages = [
        {"role": "user", "content": prompt_}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids.to(device),
        max_new_tokens=config_['max_new_tokens'],
        temperature=config_['temperature'],
        top_p=config_['top_p'],
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    data_ = {
        "prompt": prompt_,
        "response": response,
        "created_at": str(datetime.now()),
    }
    return data_


def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        config_ = {
            "top_p": float(os.getenv("TOP_P")),
            "model_path": os.getenv("MODEL_PATH"),
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS")),
            "max_works": int(os.getenv("MAX_WORKERS")),
            "temperature": float(os.getenv("TEMPERATURE"))
        }
        return config_
    except ValueError as e:
        print(f"环境变量配置错误: {e}")


def api_generation(prompts):
    # 忽略特定警告
    warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="The attention mask is not set and cannot be inferred from input because pad "
                                    "token is same as eos token.")

    device = "cuda"  # the device to load the model onto
    config = get_config()
    model_path = config.get("model_path")
    max_workers = config.get("max_works")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 多线程并发执行请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, config, prompt, tokenizer, model, device) for prompt in prompts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results
