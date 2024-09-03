import glob
import json
import os

import dotenv
from docx import Document
from dotenv import load_dotenv


# 读取docx文件，提取长文本
def read_docx(file_path_):
    doc = Document(file_path_)
    full_text = ''.join([para.text for para in doc.paragraphs])
    return full_text


# 对给定长文本进行切片
def slice_text(full_text, length, offset):
    slices = []
    start = 0
    while start < len(full_text):
        end = min(start + length, len(full_text))
        slices.append((full_text[start:end], start))
        start += offset
    return slices


# 加载上次的处理记录，获取下一次处理的相关信息
def load_record(jsonl_file):
    # 如果输出文件不存在，抛出异常
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"无法找到JSONL文件'{jsonl_file}'")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 如果文件内容为空,从第一个文件，偏移为0的位置开始, id从0开始
    if not lines:
        return 0, 0, 0

    # 读取最后一行的记录
    last_record = json.loads(lines[-1])
    last_id_ = last_record['id']
    last_file_path_ = last_record['source']
    last_offset_ = last_record['offset']

    # 加载文件列表
    docx_files_ = glob.glob(os.path.join(input_file_folder, '*.docx'))
    docx_files_.sort()
    # 确定下一个切片的id
    next_id_ = last_id_ + 1

    # 确定上次处理的文件索引
    last_file_index_ = 0
    for i_, file_path_ in enumerate(docx_files_):
        if file_path_ == last_file_path_:
            last_file_index_ = i_
            break

    # 获取当前文件字节大小
    full_text_ = read_docx(docx_files_[last_file_index_])
    file_length_ = len(full_text_)

    # 确定下一次的文件偏移位置,上次的切片偏移开始位置+切片偏移 = 下一次切片偏移起始位置
    next_offset_ = min(last_offset_ + slice_offset_unit, file_length_)

    # 当前文件是否处理完
    if file_length_ == next_offset_:
        # 当前文件已经处理完，处理下一个文件
        if last_file_index_ == len(docx_files_) - 1:
            # 所有文件已经处理完
            print('所有文件处理完毕')
            exit(1)
        else:
            # 仍有新的未处理文件
            next_file_index_ = last_file_index_ + 1
            # 偏移从0开始
            next_offset_ = 0
    else:
        # 当前文件未处理完，继续处理当前文件
        next_file_index_ = last_file_index_
    return next_id_, next_file_index_, next_offset_


def process_docx_file(file_path_, slice_length_, slice_offset_, output_file_, id_start_, start_offset_):
    # 读取文档文本
    full_text = read_docx(file_path_)

    # 对文本进行切片，从start_offset开始
    sliced_texts = slice_text(full_text[start_offset_:], slice_length_, slice_offset_)

    # 将切片结果写入 JSONL 文件
    with open(output_file_, 'a', encoding='utf-8') as f:
        for i_, (text, offset) in enumerate(sliced_texts):
            record = {
                "id": id_start_ + i_,
                "source": file_path_,
                "slice": text,
                "offset": start_offset_ + offset,
                "isLabeled": False,
                "labels": []
            }
            f.write(json.dumps(record) + '\n')
    return id_start_ + len(sliced_texts)  # 返回下一个可用的id


def get_config():
    """
    Gets configuration from environment variables.

    Returns:
    - A dictionary with configuration parameters.
    """
    try:
        dotenv.load_dotenv()
        config_ = {
            "input_file_folder": os.getenv("SLICE_GENERATION_INPUT_FOLDER"),
            "output_file": os.getenv("SLICE_GENERATION_OUTPUT_FILE"),
            "slice_length": int(os.getenv("SLICE_GENERATION_LENGTH")),
            "slice_offset_unit": int(os.getenv("SLICE_GENERATION_OFFSET_UNIT")),
        }
        return config_
    except ValueError as e_:
        print(f"环境变量配置错误: {e_}")
        exit(1)


if __name__ == '__main__':
    # 读取环境变量
    config = get_config()
    input_file_folder = config.get('input_file_folder')  # 参考文件路径
    output_file = config.get('output_file')  # 切片文件路径
    slice_length = config.get('slice_length')  # 切片大小
    slice_offset_unit = config.get('slice_offset_unit')  # 切片间隔

    # 加载已处理的文件偏移记录和最高id
    try:
        next_id, next_file_index, next_file_offset = load_record(output_file)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 获取文件夹中的所有DOCX文件
    docx_files = glob.glob(os.path.join(input_file_folder, '*.docx'))
    docx_files.sort()

    # 根据上次的记录继续处理
    for i in range(next_file_index, len(docx_files)):
        file_path = docx_files[i]
        print(f'正在处理文件: {file_path}')
        next_id = process_docx_file(file_path, slice_length, slice_offset_unit, output_file, next_id, next_file_offset)
        next_file_offset = 0
