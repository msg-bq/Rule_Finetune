import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # 我先把多线程都取消了
from typing import List, Tuple

import json

from utils.data import DatasetLoader, Example, Rationale
from utils.llm import LLM
from utils.others import move_file_to_jsonl
from utils.ExtraNameSpace import ColdStartScoreNameSpace
import utils.clean_prediction_func
import utils.score


@ColdStartScoreNameSpace.register("Example")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    pass

def llm_zero_shot_CoT(llm, input_text: str, cot_trigger: str, direct_answer_trigger_for_zeroshot_cot:str)\
        -> List[Tuple[str, str]]:
    """
    目前是直接用了autoCoT的写法
    """
    llm_input = input_text + "\n" + cot_trigger

    max_length = 4096
    rationales_answers_pair = []

    model = "gpt-3.5-turbo-1106"

    rationales = llm.generate_single_parallel(input_text=llm_input, model=model,#"gpt-4-1106-preview",
                                              temperature=0.5, topN=5) # 可以传入一个try_cnt

    for r in rationales:
        z2 = input_text + "Answer: " + r + " " + direct_answer_trigger_for_zeroshot_cot
        pred = llm.generate_single(input_text=z2, model=model, temperature=0.0)

        if pred:
            rationales_answers_pair.append((r, pred))

    return rationales_answers_pair


def zero_shot_CoT_single(args, llm: LLM, data: Example, topN: int = 1) -> (int, dict):
    print("=======================")
    x, y = data.question, data.gold_label

    print("question: ", x)

    rationales_answers_pair = llm_zero_shot_CoT(llm,
                                                input_text=x,  # 缺对LLM相关参数的控制
                                                cot_trigger=args.cot_trigger,
                                                direct_answer_trigger_for_zeroshot_cot=\
                                                args.direct_answer_trigger_for_zeroshot_cot)

    for z, pred in rationales_answers_pair:
        print("++++++++++")
        print("rationale: ", z)
        print("pred: ", pred)
        print("++++++++++")
        r = Rationale(rationale=z, prediction=pred)

        if is_high_quality_prediction(prediction=r.prediction.strip(), gold_label=y.strip()):
            data.update_rationale(r)
            print('success，√')
            break

    if not data.rationale:
        with open(os.path.join(args.data_dir, "rationale/fail.txt"), 'a') as f:
            f.write(x.strip() + '\n')
        return data

    final_dict = data.to_dict()
    save_path = os.path.join(args.data_dir, "rationale/ZeroShotCoT.jsonl")
    if not os.path.exists(save_path):
        with open(save_path, 'w'):
            pass

    with open(save_path, 'a') as f: # 如果在zero_shot_CoT存入，会有重复存入的情况
        assert len(final_dict['rationale']) == len(final_dict['prediction'])
        for r, p in zip(final_dict['rationale'], final_dict['prediction']):
            tmp_dict = {'question': final_dict['question'].strip(), 'gold_label': final_dict['gold_label'].strip(),
                        'rationale': r.strip(), 'prediction': p.strip()}
            f.write(json.dumps(tmp_dict) + '\n')

    return data

def zero_shot_CoT_single_parallel(args, llm: LLM, data: Example) -> (int, dict):
    print("=======================")
    x, y = data.question, data.gold_label

    print("question: ", x)

    rationales_answers_pair = llm_zero_shot_CoT(llm,
                                                input_text=x,  # 缺对LLM相关参数的控制
                                                cot_trigger=args.cot_trigger,
                                                direct_answer_trigger_for_zeroshot_cot=\
                                                args.direct_answer_trigger_for_zeroshot_cot)

    for z, pred in rationales_answers_pair:
        print("++++++++++")
        print("rationale: ", z)
        print("pred: ", pred)
        print("++++++++++")
        r = Rationale(rationale=z, prediction=pred)

        if is_high_quality_prediction(prediction=r.prediction.strip(), gold_label=y.strip()):
            data.update_rationale(r)
            print('success，√')
            break

    if not data.rationale:
        return data

    final_dict = data.to_dict()
    if not os.path.exists(os.path.join(args.data_dir, "rationale/parallel")):
        os.makedirs(os.path.join(args.data_dir, "rationale/parallel"))

    save_path = os.path.join(args.data_dir, f"rationale/parallel/{time.time()}.jsonl")

    with open(save_path, 'w') as f:
        assert len(final_dict['rationale']) == len(final_dict['prediction'])
        for r, p in zip(final_dict['rationale'], final_dict['prediction']):
            tmp_dict = {'question': final_dict['question'].strip(), 'gold_label': final_dict['gold_label'].strip(),
                        'rationale': r.strip(), 'prediction': p.strip()}
            f.write(json.dumps(tmp_dict) + '\n')

    return data

def zero_shot_CoT(args, llm: LLM, dataset: DatasetLoader):
    def select_dataset():
        """
        挑出没有rationale且应当获取的data
        """
        # return []
        for data in dataset:
            if data.rationale:  # 有可能rationale已经存在了，这个时候就不需要再生成了。但要注意的是，如果调整了rationale的录入格式
                # 要重载内置的__bool__函数
                continue

            if data.question.strip() in fail_examples:
                continue

            print("ZeroCoT有缺失：", data)
            yield data


    fail_examples = []
    rationale_dir = os.path.join(args.data_dir, "rationale")
    if not os.path.exists(rationale_dir):
        os.makedirs(rationale_dir)
    fail_file = os.path.join(rationale_dir, "fail.txt")
    if not os.path.exists(fail_file):
        with open(fail_file, 'w'):
            pass

    with open(fail_file, 'r') as f:
        lines = f.readlines()
        # 两行两行的遍历
        for i in range(0, len(lines), 2):
            fail_examples.append((lines[i] + lines[i + 1]).strip())

    if args.multi_thread:
        with ThreadPoolExecutor(max_workers=200) as executor:
            responses = [executor.submit(zero_shot_CoT_single_parallel, args, llm, data) for data in select_dataset()]

            for r in as_completed(responses):
                if r.result().rationale:
                    continue

                with open(os.path.join(args.data_dir, "rationale/fail.txt"), 'a') as f:
                    f.write(r.result().question.strip() + '\n')

        # 汇总保存进ZeroShotCoTParallel.jsonl
        move_file_to_jsonl(save_dir=os.path.join(args.data_dir, "rationale/parallel"),
                           save_path=os.path.join(args.data_dir, "rationale/ZeroShotCoTParallel.jsonl"))

    else:
        for data in select_dataset():
            zero_shot_CoT_single(args, llm, data)

    return dataset