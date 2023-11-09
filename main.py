from RuleFinetune.RuleTrainer import Trainer
from utils.llm import LLM, generate_func_mapping
from utils.read_datasets import read_func_mapping, read_datasets, read_rationales
import argparse


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR"],  # default包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="data dir used for experiment")

    parser.add_argument("rationale_dir", type=str, default=None,
                        help="rationale path used for experiment, name should be train/val/test.jsonl")

    parser.add_argument("--save_path", type=str, default=None,
                        help="save path used for experiment")

    parser.add_argument("--llm_model", type=str,
                        choices=["davinci", "gpt-3.5-turbo", "gpt-3.5-turbo-0613"],
                        default="gpt-3.5-turbo-0613", help="language model used for experiment")

    parser.add_argument("--llm_model_path", type=str, default=None,
                        help="language model path used for experiment")

    parser.add_argument("--multi_thread", type=bool, default=False,
                        help="whether to use multi-thread to accelerate")

    parser.add_argument("epoch", type=int, default=10,
                        help="epoch used for experiment")

    args = parser.parse_args()

    args.cot_trigger = '''A: Let's solve this problem by splitting it into steps, every step should be given by 
                       following format:\n''' \
                       '''Step n:\n''' \
                       '''Rule (Including commonsense knowledge rules): xxx(Should be concise, and give it directly)\n''' \
                       '''Premises: xxx(as the antecedent of the rule)\n''' \
                       '''Conclusion: xxx(only one conclusion)'''

    args.direct_answer_trigger_for_zeroshot_cot = "The answer is"

    return args


def main():
    """
    1. 读取数据集
    2. 构造Trainer
        2.1 构造ZeroShotCoT
        2.2 抽取出RuleBase
        2.3 进行训练
    3. 评估
    """

    args = args_parse()

    # 1. 读取数据集
    if args.rationale_dir:
        train_dataset, valid_dataset, test_dataset = read_rationales(args)
    else:
        train_dataset, valid_dataset, test_dataset = read_datasets(args)

    # 2. 构造Trainer
    # 2.1 构造ZeroShotCoT + # 2.2 抽取出RuleBase
    generate_func = generate_func_mapping[args.llm_model]
    llm_model = LLM(generate_func)

    cur_Trainer = Trainer(args, train_dataset, valid_dataset, test_dataset, llm_model)
    cur_Trainer.cold_start()

    # 2.3 进行训练
    cur_Trainer.train()

    # 3. 评估
    cur_Trainer.eval()

if __name__ == '__main__':
    main()
    