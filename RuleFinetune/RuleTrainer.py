import os.path
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from RuleFinetune.autoCoT import demo_cluster, llm_n_shot_CoT, n_shot_prompt, DemoBaseMAB, Demo
from RuleFinetune.cold_start_func import zero_shot_CoT
from utils.data import RuleBase, DatasetLoader, Example, Rationale
from utils.llm import LLM
import Levenshtein

from logger import logger


class Trainer:
    def __init__(self, args, train_dataset: DatasetLoader, valid_dataset: DatasetLoader, test_dataset: DatasetLoader,
                 llm: LLM, rule_base: RuleBase = RuleBase()):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.llm = llm
        self.rule_base = rule_base

        self.max_epoch = args.epoch

        self.lock = threading.Lock()

    def cold_start(self):
        """
        计划从这里调取第一次的CoT
        """
        dataset = zero_shot_CoT(self.args, self.llm, self.train_dataset)
        for data in dataset:
            rules = []
            for r in data.rationale:
                rules += r.extract_rules_cold_start()  # 这儿也没有根据prediction和label的一致性选择正确的rule
            self.rule_base._add_rules(rules, data.question)

        self.rule_base.save(f"./data/{self.args.dataset}/rule_base_cold")
        logger.info("完成cold start")

    def forward(self, example, demos, added_rules):
        _, response = llm_n_shot_CoT(self.llm, added_rules, input_text=example.question, demos=demos,
                                     temperature=0.3)
        print("response:\n", response)
        new_rationale = example.parse_response(response, self.args)
        if new_rationale['rationale'] != "" and new_rationale['prediction'] != "":
            print('++++' * 50)
            print("new_rationale:\n", new_rationale)
            example.update_rationale(new_rationale)  # 做了inplace的更新

            return example.rationale[-1]

        return None

    def score(self, pred: str, gold: str) -> float:
        """
        比对prediction和gold_label打分，用于调整Rule confidence
        """
        edit_distance = Levenshtein.distance(pred, gold)
        if edit_distance == 0:
            score = 1
        else:
            score = 1 - 2 * edit_distance / max(len(pred), len(gold))
        return score

    def backward(self, example: Example, added_rules: str, bandit: DemoBaseMAB, score: float, k: List[int]):
        rationale = example.rationale[-1]
        occured_rules = rationale.extract_rules_training()
        print("occured_rules:", occured_rules)
        # 这边需要一步处理，因为new_rules: List[List[str]]，（可能需要去重），最终要变成List[str]
        self.rule_base.update_rule(added_rules, occured_rules, example, score)
        print('oooo' * 50)

        # with self.lock:
        # loss取值[-1,1]，以概率形式加入到bandit里面
        prob = (score + 1) / 2
        if prob > random.random():
            demo: Demo = Demo(question=example.question,
                              rationale=example.rationale[-1].rationale,
                              gold_label=example.gold_label)
            bandit.add_demo(demo)

        loss_bandit = (score + 1) / 2
        bandit.backward(k, loss_bandit)  # 考虑下这个r对于_a和regret之类的是不是一个数

    # def backward(self, example, added_rules: str):
    #     """
    #     input: Rule_base, rationale, score
    #     output: None (RuleBase updated 如果有规则就更新，否则就加入)
    #     """
    #     rationale = example.rationale[-1]
    #     new_rules = rationale.extract_rules_training()
    #     self.rule_base.update_q2r(example.question, new_rules)
    #     print("new_rules:", new_rules)
    #     # 这边需要一步处理，因为new_rules: List[List[str]]，（可能需要去重），最终要变成List[str]
    #     self.rule_base.update_rule(added_rules, new_rules, rationale.prediction, example.gold_label)
    #     print('oooo'*50)

    def train_step(self, example: Example, bandit: DemoBaseMAB):
        k, demos = bandit.sample_demos(topk=5)

        demos, added_rules = n_shot_prompt(self.args,
                                           rules=self.rule_base.sample_rules(
                                               do_not_use_question=example.question
                                           ),
                                           demos=demos)
        if example.question in demos:  # 作为样例的题就没必要forward了
            return

        rationale = self.forward(example, demos, added_rules)

        if rationale:
            score = self.score(rationale.prediction, example.gold_label)
            with self.lock:
                self.backward(example, added_rules, bandit, score, k)
        else:
            score = -1

        return score
        # self.backward(example, added_rules)

        # with open(added_rules_save_path, 'w', encoding="utf8") as f:
        #     f.write(str(added_rules))

    def train(self):
        bandits, question2cluster = demo_cluster(self.args, self.train_dataset)
        # 这个地方还不好替换成别的方案，封装的不是特别特别充分
        save_path = f"{self.args.save_dir}/train_loss.txt"
        with open(save_path, 'w', encoding="utf8") as f:
            pass

        for ep in range(self.max_epoch):  # 这里最好是epoch
            self.force_write(ep)

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(self.train_step, example, bandits[question2cluster[example.question]])
                           for example in self.train_dataset]

                losses = [future.result() for future in futures if future.result() and future.result() != -1]
                losses = [(l + 1) / 2 for l in losses]
                # None对应样例、-1对应输出没有rationale的样例
                logger.info(f"epoch{ep}的平均score为：{sum(losses) / len(losses)}") # 如果像正常的微调

                with open(save_path, 'a', encoding="utf8") as f:
                    f.write(f"epoch{ep}的平均score为：{sum(losses) / len(losses)}\n")
                # 其实训练集的信息是会被过拟合记住的，所以那个要求sample rule的时候不能用来源question的规则
                # 这条限制，是可以保留或者说可控的。这种过拟合也比参数微调方便控制

            # 保存self.rule_base._rule_name_2_rule_instance
            self.rule_base.save(f"{self.args.save_dir}/rule_base_epoch{ep}")
            # 保存bandits
            for k, v in bandits.items():
                v.save(f"{self.args.save_dir}/bandits_epoch{ep}_{k}")

            self.rule_base.broadcast_rule_info() # 每个epoch统一平均，避免并行带来的不同步

        self.rule_base.save(f"{self.args.save_dir}/rule_base_final")

    def eval(self):
        """
        将模型置于evaluate模式，包括：# 其实某种意义上，可以说读入rule_base应该专门有一个load函数，而不是放在这里。train.rationale可能也应该在load里读入
        1. 将llm置于evaluate模式
        2. 读入rule_base
        3. 读入train_dataset的rationale，并生成demos。由于此刻train里面还没保存，所以就先读入demos_epoch{epoch-1}作为替代
        """
        # 1. 将llm置于evaluate模式
        # self.llm.eval() 这行代码暂时还无法生效

        # 2. 读入rule_base
        if len(self.rule_base) == 0:  # 这个我觉得是，只有纯测试的时候才需要读入。平时的话规则库本来就在训练过程中有了
            # 如果还想考虑一个特殊情况的话，就是checkpoint。但是这个我觉得也没必要，因为checkpoint的load阶段就应该读入了
            with open(f"{self.args.save_dir}/rule_base_final", encoding="utf8") as f:
                rules = [l for l in f.readlines() if l.strip()]
                self.rule_base.read_rules(rules)

        # 3. 读入train_dataset的rationale
        # 目前还没保存训练集，所以直接读取demos和added_rules
        with open(f"{self.args.save_dir}/demos_epoch{0}", encoding="utf8") as f:
            self.eval_demos = "".join([l for l in f.readlines()])

        save_path = f"{self.args.save_dir}/demos_eval"
        with open(save_path, 'w', encoding="utf8") as f:
            f.write(str(self.eval_demos))

        with open(f"{self.args.save_dir}/added_rule_epoch{0}", encoding="utf8") as f:
            self.eval_added_rules = "".join([l for l in f.readlines()])

    def eval_step(self, example: Example):
        # 预留一个后处理demos的函数，是hjq写的

        _, response = llm_n_shot_CoT(self.llm, self.eval_added_rules,
                                     input_text=example.question, demos=self.eval_demos)
        rationale = example.parse_response(response, self.args)
        prediction = Rationale.clean_prediction(rationale['prediction'])
        print(rationale['prediction'])
        print(prediction, example.gold_label)
        return prediction, example.gold_label

    def evaluate(self, is_valid=False, special_datasets: DatasetLoader = None):
        """
        验证集和测试集的评估
        """
        eval_type = "valid" if is_valid else "test"
        datasets = special_datasets if special_datasets else self.valid_dataset if is_valid else self.test_dataset

        # if is_valid: # valid用于训练阶段的测试
        #     demos = demo_cluster(self.args, self.train_dataset) #目前这个手段，还不能做到逐阶段优化5-shot
        #     # 因为demo_cluster里用的rationale，只是random.choice了一个rationale，而不是根据score来选
        #     demos, added_rules = n_shot_prompt(self.args, rules=self.rule_base.sample_rules(), demos=demos)
        # else:
        #     pass

        correct_cnt = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self.eval_step, example) for example in datasets]
            for future in futures:
                prediction, gold_label = future.result()
                if prediction == gold_label:
                    correct_cnt += 1

        logger.info(f"{eval_type}集上的准确率为：{correct_cnt / len(datasets)}")

    def test(self):
        pass

    def force_write(self, ep):
        save_path = f"{self.args.save_dir}/rule_base_epoch{ep}"
        added_rules_save_path = f"{self.args.save_dir}/added_rule_epoch{ep}"
        demos_save_path = f"{self.args.save_dir}/demos_epoch{ep}"
        for p in [save_path, added_rules_save_path, demos_save_path]:
            if os.path.exists(p):
                os.remove(p)

# if __name__ == '__main__':
#     r = Rationale('a', ['b'], ['c'], 'd')
#     rb = RuleBase()
#     rb.update_rule('rule', 1)
#     rb.update_rule('rule', 2)
#     rb.update_rule('rule3', 3)
#     print(rb.write_rules())
