import os.path

from RuleFinetune.RuleTrainer import Trainer
from utils.llm import LLM, generate_func_mapping
from utils.read_datasets import read_datasets, read_rationales
import argparse
from utils.ExtraNameSpace import NameSpace

from prompt import cot_trigger, pred_trigger


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR", "STS-B", "LANG-8"],  # default包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="data dir used for experiment")

    parser.add_argument("--rationale_dir", type=str, default=None,
                        help="rationale path used for experiment, name should be train/val/test.jsonl")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="save dir used for experiment")

    parser.add_argument("--llm_model", type=str,
                        choices=["davinci", "gpt-3.5-turbo", "gpt-3.5-turbo-0613",
                                 "gpt-3.5-turbo-1106"],
                        default="gpt-3.5-turbo-1106", help="language model used for experiment")

    parser.add_argument("--llm_model_path", type=str, default=None,
                        help="language model path used for experiment")

    parser.add_argument("--multi_thread", type=bool, default=False,
                        help="whether to use multi-thread to accelerate")

    parser.add_argument("--epoch", type=int, default=3,
                        help="epoch used for experiment")

    parser.add_argument("--topN", type=int, default=1,
                        help="output topN results in every call LLM.generate")

    parser.add_argument("--train", type=bool, default=True,
                        help="whether to train")

    parser.add_argument("--eval", type=bool, default=False,
                        help="whether to eval")

    parser.add_argument("--test", type=bool, default=False,
                        help="whether to test")

    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode")  # 这个参数源于autoCoT

    parser.add_argument("--random_seed", type=int, default=192, help="random seed")  # 这个参数源于autoCoT

    parser.add_argument("--num_clusters", type=int, default=5,
                        help="the number of clusters for questions")

    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    ) #源自autoCoT

    parser.add_argument(
        "--demo_save_dir", type=str, default='demosave', help="save dir for demo examples"
    ) #源自autoCoT

    parser.add_argument("--cot_trigger_type", type=str, default='default1',
                        choices=['default1', 'default2', 'HtT'],
                        help="cot trigger type")

    args = parser.parse_args()

    args.cot_trigger = cot_trigger[args.cot_trigger_type]
    args.pred_trigger = pred_trigger[args.dataset]

    args.direct_answer_trigger_for_zeroshot_cot = args.pred_trigger

    NameSpace._args = args

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
    if not args.data_dir:
        args.data_dir = f"./data/{args.dataset}"

    # 1. 读取数据集
    train_dataset, valid_dataset, test_dataset = read_datasets(args)

    if args.multi_thread:
        if os.path.exists(os.path.join(args.data_dir, "rationale/ZeroShotCoTParallel.jsonl")):
            args.rationale_dir = os.path.join(args.data_dir, "rationale/ZeroShotCoTParallel.jsonl")
    else:
        if os.path.exists(os.path.join(args.data_dir, "rationale/ZeroShotCoT.jsonl")):
            args.rationale_dir = os.path.join(args.data_dir, "rationale/ZeroShotCoT.jsonl")

    if args.rationale_dir:
        train_dataset, valid_dataset, test_dataset = read_rationales(args,
                                                                     train_dataset=train_dataset,
                                                                     valid_dataset=valid_dataset,
                                                                     test_dataset=test_dataset)

    # 2. 构造Trainer
    # 2.1 构造ZeroShotCoT + # 2.2 抽取出RuleBase
    generate_func = generate_func_mapping[args.llm_model]
    llm_model = LLM(generate_func)

    cur_Trainer = Trainer(args, train_dataset, valid_dataset, test_dataset, llm_model) #topN是个小问题

    if args.train:    # 需要cold start的时候运行
        cur_Trainer.cold_start()  # 存Answer的时候就clean一下

    # 2.3 进行训练
    if args.train:
        cur_Trainer.train()

    # # 3. 评估

    # if args.eval:
    #     cur_Trainer.eval()
    #     cur_Trainer.evaluate(is_valid=True)
    #
    # if args.test:
    #     cur_Trainer.eval()
    #     cur_Trainer.evaluate(is_valid=False)


if __name__ == '__main__':
    main()
    """
    --dataset
    CLUTRR
    --data_dir
    C:/Users/lbq/Documents/GitHub/Rule_Finetune/data/CLUTRR
    --save_dir
    C:/Users/lbq/Documents/GitHub/Rule_Finetune/experiment
    --llm_model
    gpt-3.5-turbo
    """
    # data_dir = r"D:\DATA\lbq\git\Rule_Finetune\data\CLUTRR"
    # if not os.path.exists(os.path.join(data_dir, "rationale/ZeroShotCoTParallel.jsonl")):
    #     with open(os.path.join(data_dir, "rationale/ZeroShotCoTParallel.jsonl"), 'w') as f:
    #         pass
    #
    # save_dir = os.path.join(data_dir, "rationale/parallel")
    #
    # for file in os.listdir(save_dir):
    #     path = os.path.join(save_dir, file)
    #     with open(path, 'r') as f:
    #         lines = f.readlines()
    #         with open(os.path.join(data_dir, "rationale/ZeroShotCoTParallel.jsonl"), 'a') as f:
    #             for line in lines:
    #                 if line.strip():
    #                     f.write(line)
    #
    #     os.remove(path)
