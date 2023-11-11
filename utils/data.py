import math
import re
import string
from typing import List, Dict, Optional, Union, Tuple
import Levenshtein
import pandas as pd


class Rule:
    def __init__(self, content: str, confidence: float = 0.0):
        self.content = content
        self.confidence = confidence


class RuleBase:
    def __init__(self):
        self.rules = pd.DataFrame({'rule_name': [],
                                   'rule_instance': []})

    def write_rules(self) -> str:
        """
        返回所有rules作为prompt
        TODO: 这是简易版本
        """
        out = '\n'.join([row['rule_name'] for row_idx, row in self.rules.iterrows()])
        return out

    def update_rule(self, rule_names: List[str], scores: Optional[List[float]] = None) -> List[Rule]:
        """
        需要字符串匹配，找到就返回，找不到就创建+返回
        TODO: 这是直接匹配的版本
        """
        rule_instances = []
        if not scores:
            scores = [1.0] * len(rule_names)

        for rule_name, score in zip(rule_names, scores):
            matched_rules = self.rules.loc[(self.rules["rule_name"] == rule_name)]
            assert len(matched_rules) <= 1
            if len(matched_rules) == 0:
                new_rule_instance = self._add_rules([rule_name], [score])[0]
                rule_instances.append(new_rule_instance)
            elif len(matched_rules) == 1:
                # TODO: score是覆写还是加上
                idx = self.rules[self.rules.rule_name == rule_name].index.tolist()[0]
                rule_instance = self.rules.iloc[idx]['rule_instance']
                rule_instance.confidence = score
                rule_instances.append(rule_instance)
        return rule_instances

    def _add_rules(self, rules: List[str], scores: List[float]) -> List[Rule]:
        """
        这里需要一个添加rule的函数，包括将字符串转为str+查重+添加
        """
        new_rule_instances = []
        for rule_name, score in zip(rules, scores):
            new_rule_instance = Rule(content=rule_name, confidence=score)
            new_rule_df = pd.DataFrame({'rule_name': [rule_name],
                                        'rule_instance': [new_rule_instance]})
            self.rules = pd.concat([self.rules, new_rule_df], ignore_index=True)
            new_rule_instances.append(new_rule_instance)
        return new_rule_instances


class Rationale:    # 修正到只有两个属性
    """
    top-N的结果，rationale和prediction
    """

    def __init__(self, rationale: str, prediction: str):
        self.rationale = rationale
        self.prediction = prediction

    def extract_rules(self, rationale: str) -> List[str]:
        """
        从dataset中抽取出rules
        """
        rule_pattern = re.compile(r"Rule:(.+?)(?:\.|\n)")
        rules = rule_pattern.findall(rationale)
        rules = [r.strip() for r in rules]

        return rules

    def clean_prediction(self, prediction: str) -> str:
        """
        从Answer中提出特定数据集的答案，可能需要根据数据集的不同进行修改
        这里传入的pred已经是最后一个The answer is后面的部分了
        """
        pred_words = prediction.split()
        if len(pred_words) == 1:
            if pred_words[0][-1] in string.punctuation:
                return pred_words[0][:-1]

            return pred_words[0]

        if pred_words[-1][-1] in string.punctuation:
            return pred_words[-1][:-1]

        return pred_words[-1]

    # # 因为gold_label无法获取而搁置，可能放到其它地方
    # def score(self) -> float:
    #     """
    #     比对prediction和gold_label打分，用于调整Rule confidence
    #     TODO: 实现了一个简易的负对数编辑距离打分
    #     """
    #     edit_distance = [Levenshtein.distance(p, self.gold_label) for p in self.prediction]
    #     scores = [-math.log(ed) for ed in edit_distance]
    #     score = sum(scores)
    #     return score

    def update(self, new_rationale: Dict[str, str]):
        """
        lbq
        做对了更新覆盖，错了不变
        """
        for key in new_rationale.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = new_rationale[key]
            else:
                raise KeyError("The key is not in the DatasetLoader")

        return self

    def __eq__(self, other):
        if isinstance(other, Rationale):
            return self.__dict__ == other.__dict__
        else:
            raise AttributeError("Incorrect attribute!")


class Example:
    def __init__(self, question: str, gold_label: str, rationale: List[Rationale] = None):
        self.question = question
        self.gold_label = gold_label
        self.rationale = [] if rationale is None else rationale

    def __eq__(self, other):
        if isinstance(other, Example):
            return self.__dict__ == other.__dict__
        else:
            raise AttributeError("Incorrect attribute!")

    def _merge_arrtibute(self, attr: str, other_attr_value: Union[List, str]):
        """
        合并两个属性，这里只是简单的合并，不做去重
        """
        if getattr(self, attr, None) is None:
            raise AttributeError("The attribute is not in the Example")

        if isinstance(getattr(self, attr), list):
            if isinstance(other_attr_value, list):
                getattr(self, attr).extend(other_attr_value)
            elif isinstance(other_attr_value, str):
                getattr(self, attr).append(other_attr_value)
            else:
                raise TypeError("Incorrect type.")

        elif isinstance(getattr(self, attr), str):
            if isinstance(other_attr_value, str):
                 setattr(self, attr, other_attr_value)
            else:
                raise TypeError("Incorrect type.")

    def _check_QA(self, other_QA: Tuple[str, str]):
        if self.question and self.question != other_QA[0]:
            raise Warning("The question is not the same.")

        if self.gold_label and self.gold_label != other_QA[1]:
            raise Warning("The gold_label is not the same.")

    def update(self, example: [str, Dict[str, str], 'Example']):
        """
        根据example里面的更新self对应的值。请注意，这个函数用于更新某个样例的情况，而不建议将样例1改变为样例2（注意到
        list我们是直接append，而不是替换的）
        当待修改的question和gold_label有任一不同时，会抛出警告
        """
        merge_example = example

        if isinstance(example, str): #parse出来一个dict
            self.parse_response(example)
        elif isinstance(example, Example):
            merge_example = example.to_dict()

        if isinstance(merge_example, dict):
            self._check_QA((merge_example.get('question', None), merge_example.get('gold_label', None)))

            for key in merge_example.keys():
                if key in self.__dict__:
                    self._merge_arrtibute(key, merge_example[key])
                else:
                    raise KeyError("The key is not in the DatasetLoader")

        else:
            raise TypeError("Incorrect type.")

        return self

    def to_dict(self):
        return self.__dict__

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        这里只会传入A：后面生成的部分
        """
        response_lst = response.lower().split("the answer is")
        length = len(response)
        answer = response[length-len(response_lst[-1])].strip()
        rationale = response[:-(len(response_lst[-1]) + len("the answer is"))].strip()

        return {'question': self.question, 'rationale': rationale, 'gold_label': answer}

    def __repr__(self):
        pass

    def __hash__(self):
        return hash((self.question, self.gold_label))

class DatasetLoader:  # 命名上和torch的多加了个set
    def __init__(self, data: List[Example]):
        self.data = pd.DataFrame({'data_question': [],
                                  'data_gold_label': [],
                                  'data_instance': []})
        for e in data:
            self.data = pd.concat([data, pd.DataFrame({'data_question': [e.question],
                                                       'data_gold_label': [e.gold_label],
                                                       'data_instance': [e]})], ignore_index=True)


    def __repr__(self):
        print(" ".join([d for d in self.data]))