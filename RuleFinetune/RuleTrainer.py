from typing import List

from RuleFinetune.autoCoT import demo_cluster, llm_n_shot_CoT
from RuleFinetune.cold_start_func import zero_shot_CoT
from utils.data import RuleBase, DatasetLoader
from utils.llm import LLM


class Trainer:
    def __init__(self, args, train_dataset: DatasetLoader, valid_dataset: DatasetLoader, test_dataset: DatasetLoader,
                 llm: LLM, rule_base: RuleBase = RuleBase()):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.llm = llm
        self.rule_base = rule_base

        self.max_iterations = args.epoch #这个可以调掉

    def cold_start(self):
        """
        计划从这里调取第一次的CoT
        """
        dataset = zero_shot_CoT(self.llm, self.train_dataset)
        for data in dataset.data:
            rules = self.extract_rules(data.rationale)
            self.rule_base.add_rules(rules)


    def forward(self):
        """
        # first zero-shot
        for R in TrainDataset:
            prompt = Rule_base.write_rules + 5_shot(R.q, TrainDataset) + R.q
            response = call_gpt(prompt)
            rationale = parse_response(response)
            R.update(rationale)
            score = rationale.score()
        # first 5-shot
        for R in

        """
        for it in range(self.max_iterations): # 这里最好是epoch
            demos = demo_cluster(self.args, self.train_dataset)  # 每一轮的开头，构造本轮所使用的n-shot CoT

            for rationale in self.train_dataset:
                prompt = self.rule_base.write_rules + '\n' + \
                         self.cold_start(rationale.question, self.train_dataset) + '\n' + \
                         rationale.question
                response = llm_n_shot_CoT(args=self.args, llm=self.llm, prompt=prompt, demos=demos)
                new_rationale = parse_response(response)
                rationale.update(new_rationale)     # 做了inplace的更新，所以train_dataset无需更新
                score = rationale.score()
                new_rules = rationale.extract_rules()
                # 这边需要一步处理，因为new_rules: List[List[str]]，（可能需要去重），最终要变成List[str]
                [self.rule_base.update_rule(rule, score) for rule in new_rules]

                # （划掉）然后是对train dataset的更新或新创建

    def backward(self):
        """
        input: Rule_base, rationale, score
        output: None (RuleBase updated 如果有规则就更新，否则就加入)
        """
        pass    # 部分写进forward里了，forward、backward是否需要分离，不知道这里需要写上面


    def train(self):
        """
        1. cold start
        for _ in range(epoch):
            2. forward
            3. backward
            (可能还包括验证集)
        """
        # for _ in range(epoch):
        #     demos = demo_cluster(train_dataset) # 每一轮的开头，构造本轮所使用的n-shot CoT
        pass

    def eval(self):
        """
        指验证集，但可以和test合并
        """
        pass

    def test(self):
        pass

if __name__ == '__main__':
    r = Rationale('a', ['b'], ['c'], 'd')
    rb = RuleBase()
    rb.update_rule('rule', 1)
    rb.update_rule('rule', 2)
    rb.update_rule('rule3', 3)
    print(rb.write_rules())