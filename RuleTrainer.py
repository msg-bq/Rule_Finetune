class Trainer:
    def __init__(self, train_dataset, rule_base):
        self.train_dataset = train_dataset
        self.rule_base = rule_base

        self.max_iterations = 10

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
        for it in range(self.max_iterations):
            for rationale in self.train_dataset:
                prompt = self.rule_base.write_rules + '\n' + \
                         n_shot(rationale.question, self.train_dataset) + '\n' + \
                         rationale.question
                response = call_gpt(prompt)
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

if __name__ == '__main__':
    r = Rationale('a', ['b'], ['c'], 'd')
    rb = RuleBase()
    rb.update_rule('rule', 1)
    rb.update_rule('rule', 2)
    rb.update_rule('rule3', 3)
    print(rb.write_rules())
