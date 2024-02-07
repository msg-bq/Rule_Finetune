import argparse
import operator
import os
import re
from concurrent.futures import ThreadPoolExecutor

from baseline.rule_sample import sample_rule_strategy
from utils.data import Rationale, Example, RuleBase, DisjointSetRuleBase
from utils.llm_models.call_openai import call_openai
from utils.read_datasets import read_datasets

from prompts import dataset_prompt
from utils.sub_score import parse_response

import utils.clean_prediction_func

# import utils.sub_score as score_py

import utils.ExtraNameSpace as ExtraNameSpace

parser = argparse.ArgumentParser(description="Rule-Finetune")

parser.add_argument("--dataset", type=str, default="LANG_8",
                        choices=["default", "CLUTRR", "STS_B", "LANG_8", "CLUTRR_textual"],  # default包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")
parser.add_argument("--data_dir", type=str, default=None,
                        help="data dir used for experiment")

parser.add_argument("--prompt_type", type=str, default="CoT_rule",
                        choices=["zero-shot", "CoT", "CoT_rule", "CoT_HtT"],
                        help="prompt type used for experiment")

parser.add_argument("--sample_strategy", type=str, default="confidence_1800", #"confidence_num"
                    help="sample rule strategy used for experiment")

parser.add_argument("--rule_base_path", type=str, default="../experiment/rule_base_final",
                        help="rule base path used for experiment")

parser.add_argument("--dataset_type", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="dataset type used for experiment")

parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613",
                        choices=["gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-3.5-turbo",
                                 "gpt-4-1106-preview"],
                        help="model used for experiment")

args = parser.parse_args()

if not args.data_dir:
    args.data_dir = "../data/" + args.dataset

ExtraNameSpace.NameSpace._args = args

train_dataset, valid_dataset, test_dataset = read_datasets(args)

rule_base = DisjointSetRuleBase()
args.rule_base_path = '../experiment/LANG_8/version_2/rule_base_epoch4'
rule_base.read_rules(args.rule_base_path)

dir_path = f"./{args.model}/{args.dataset}/{args.prompt_type}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

num_suffix = 0
while os.path.exists(f"{dir_path}/{args.sample_strategy}_{num_suffix}.txt"):
    num_suffix += 1
config_save_path = f"{dir_path}/{args.sample_strategy}_config_{num_suffix}"

while os.path.exists(f"{dir_path}/{args.sample_strategy}_{num_suffix}.txt"):
    num_suffix += 1
save_path = f"{dir_path}/{args.sample_strategy}_{num_suffix}.txt"

with open(config_save_path, 'w', encoding="utf8") as f:
    f.write(str(args)+'\n')

    if args.prompt_type == "zero-shot":
        pass
    elif args.prompt_type == "CoT":
        f.write(dataset_prompt[args.dataset]['CoT'])
    elif args.prompt_type == "CoT_rule":
        f.write(dataset_prompt[args.dataset]['rule_instruction'])
        f.write(dataset_prompt[args.dataset]['CoT_rule'])
    elif args.prompt_type == "CoT_HtT":
        f.write(dataset_prompt[args.dataset]['rule_instruction_HtT'])
        f.write(dataset_prompt[args.dataset]['CoT_HtT'])
    else:
        raise NotImplementedError("prompt type not implemented")

    f.write("\n")

    if args.sample_strategy.startswith("confidence"):
        f.write(f"max_confidence_num: {args.sample_strategy.split('_')[1]}")
        f.write("\n")
        f.write("rule_base:\n")
        for rule in rule_base._rule_name_2_rule_instance.values():
            f.write(rule.content + '\n')
    else:
        raise NotImplementedError("sample strategy not implemented")

with open(save_path, 'w', encoding="utf8") as f:
    pass

def eval_step(args, example: Example):
    global rule_base

    if args.prompt_type == "zero-shot":
        prompt = example.question + "\nAnswer:"
    elif args.prompt_type == "CoT":
        prompt = dataset_prompt[args.dataset]['CoT'] + example.question + "\nAnswer:"
    else:
        if args.sample_strategy.startswith("confidence"):
            max_confidence_num = int(args.sample_strategy.split("_")[1])
            sampled_rules = sample_rule_strategy['confidence'](rule_base, max_confidence_num)
        else:
            raise NotImplementedError("sample strategy not implemented")

        added_rules = '\n'.join([
            # str(idx+1)+':\t'+rn.content
            rn.content
            for idx, rn in enumerate(sampled_rules)])

        if args.prompt_type == "CoT_rule":
            prompt = dataset_prompt[args.dataset]['rule_instruction'] + added_rules + \
                     example.question + "\nAnswer:"  #+ dataset_prompt[args.dataset]['CoT_rule'] + \

        elif args.prompt_type == "CoT_HtT":
            prompt = dataset_prompt[args.dataset]['rule_instruction_HtT'] + '\n' + dataset_prompt[args.dataset]['CoT_HtT'] + \
                     example.question + "\nAnswer:"
        else:
            raise NotImplementedError("prompt type not implemented")

    response = call_openai(prompt, model=args.model)
    rationale = example.parse_response(response)
    prediction = Rationale.clean_prediction(rationale['prediction'])

    # try:
    #     score = parse_response(question=example.question, response=response,
    #                        added_rules="\n".join([rn.content for rn in sampled_rules]))
    # except:
    #     score = 0
    score = 0
    print(rationale, prediction, example, score, )
    return rationale, prediction, example, score

correct_cnt = 0
scores = []

if args.dataset_type == "train":
    final_dataset = train_dataset
elif args.dataset_type == "valid":
    final_dataset = valid_dataset
elif args.dataset_type == "test":
    final_dataset = test_dataset

with ThreadPoolExecutor(max_workers=200) as executor:
    futures = [executor.submit(eval_step, args, example) for example in final_dataset]
    for future in futures:
        rationale, prediction, example, score = future.result()
        gold_label = example.gold_label
        if args.dataset == "LANG_8":
            prediction = prediction.strip().replace(' ', '')
            gold_label = gold_label.strip().replace(' ', '')
            if prediction == "GOLD_LABEL":
                pattern = "Sentence: (.*)\nQuestion: What's the grammar errors and revised sentence of above sentence?"
                prediction = re.match(pattern, rationale).group(1).strip().replace(' ', '')

        if prediction.lower() == gold_label.lower():
            print("====================")
            print("这个样例做对了")
            print("rationale:", rationale)
            print("prediction:", prediction)
            print("gold_label:", gold_label)

            correct_cnt += 1

        with open(save_path, 'a', encoding="utf8") as f:
            f.write(str(rationale) + '\n')

        scores.append(score)

accuracy = correct_cnt / len(final_dataset)
print(f"测试集上的准确率为：{accuracy}")
with open(config_save_path, 'a', encoding="utf8") as f:
    f.write(f"测试集上的准确率为：{accuracy}\n")

# print("=====================")
# for key in ["step_relation_path", "length_reduce", "first_two_relations", "rule_retrieve", "step_inference_accuracy"]:
#     success_key_score = [score[key] for score in scores if key in score]
#     s = []
#     for i in success_key_score:
#         s.extend(i)
#
#     print(key, sum(s)/len(s))
#     if len(success_key_score):
#         print(key, sum(success_key_score)/len(success_key_score))