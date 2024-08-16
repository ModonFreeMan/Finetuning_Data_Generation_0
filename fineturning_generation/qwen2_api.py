import os
import dotenv
from torch import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


def make_request(config_, prompt_, device):
    model = AutoModelForCausalLM.from_pretrained(config_['model_path'], torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(config_['model_path'])

    messages = [{"role": "user", "content": prompt_}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=config_['max_new_tokens'])
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    data_ = {"prompt": prompt_, "response": response, "created_at": str(datetime.now())}
    return data_


def worker(config, prompts, device):
    results = []
    for prompt in prompts:
        result = make_request(config, prompt, device)
        results.append(result)
    return results


def get_config():
    dotenv.load_dotenv()
    return {
        "model_path": os.getenv("MODEL_PATH"),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS")),
        "device": os.getenv("DEVICE"),
        "max_workers": int(os.getenv("MAX_WORKERS")),
    }


def api_generation(prompts):
    config = get_config()
    device = config.get('device')
    num_processes = config.get('max_workers')  # 进程数

    chunked_prompts = [prompts[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(worker, [(config, chunk, device) for chunk in chunked_prompts])

    return [item for sublist in results for item in sublist]
