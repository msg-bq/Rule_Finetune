import logging
import os.path
from concurrent.futures import ThreadPoolExecutor #我先把多线程都取消了
from typing import List
import json

import numpy as np

import threading
import json

from utils.data import DatasetLoader, Example
from utils.llm import LLM


def llm_zero_shot_CoT(llm, input_text: str, cot_trigger: str, direct_answer_trigger_for_zeroshot_cot:str):
    """
    目前是直接用了autoCoT的写法
    """
    input_text = input_text + " " + cot_trigger

    max_length = 2048
    z = llm.generate_single(input_text=input_text, temperature=0.3, max_length=max_length)

    z2 = input_text + z + " " + direct_answer_trigger_for_zeroshot_cot
    pred = llm.generate_single(input_text=z2, temperature=0.0, max_length=max_length)

    return z, pred


def zero_shot_CoT_single(args, llm: LLM, data: Example, topN: int = 1) -> (int, dict):
    x, y = data.question, data.gold_label

    z, pred = llm_zero_shot_CoT(llm,
                                input_text=x, # 缺对LLM相关参数的控制
                                cot_trigger=args.cot_trigger,
                                direct_answer_trigger_for_zeroshot_cot=args.direct_answer_trigger_for_zeroshot_cot)

    data.rationale = z
    data.prediction = pred

    final_dict = data.to_dict()
    save_path = os.path.join(args.save_dir, "rationale/ZeroShotCoT.jsonl")
    if not os.path.exists(save_path):
        with open(save_path, 'w'):
            pass

    with open(save_path, 'a') as f: # 如果在zero_shot_CoT存入，会有重复存入的情况
        f.write(json.dumps(final_dict) + '\n')

    return data


def zero_shot_CoT(args, llm: LLM, dataset: DatasetLoader):
    if args.multi_thread:
        pass
    else:
        for data in dataset.data:
            if data.rationale:  # 有可能rationale已经存在了，这个时候就不需要再生成了。但要注意的是，如果调整了rationale的录入格式
                # 要重载内置的__bool__函数
                continue

            zero_shot_CoT_single(args, llm, data)

    return dataset
