import argparse


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR"],  # default包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")

    parser.add_argument("--train_path", type=str, default=None,
                        help="train dataset path used for experiment")

    parser.add_argument("--valid_path", type=str, default=None,
                        help="valid dataset path used for experiment")

    parser.add_argument("--test_path", type=str, default=None,
                        help="test dataset path used for experiment")

    parser.add_argument("--llm_model", type=str,
                        choices=["davinci", "gpt-3.5-turbo", "gpt-3.5-turbo-0613"],
                        default="gpt-3.5-turbo-0613", help="language model used for experiment")

    parser.add_argument("--llm_model_path", type=str, default=None,
                        help="language model path used for experiment")

    return parser.parse_args()





def main():
    """
    1. 读取数据集
    2. 构造ZeroShotCoT
    3. 抽取出RuleBase
    4. 构造Trainer
    5. 进行训练
    """

    args = args_parse()

    # 1. 读取数据集

