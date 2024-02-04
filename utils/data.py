import math
import re
import string
from collections import defaultdict
from random import choice, choices
from typing import List, Dict, Optional, Union, Tuple, overload, Set
import Levenshtein
import numpy as np
import pandas as pd
import operator
import random

from sentence_transformers import SentenceTransformer

from utils.ExtraNameSpace import PredictionCleanNameSpace, RuleExtractionNameSpace
import utils.extract_rule


class Rule:
    def __init__(self, content: str, question: str, confidence: float = 0.0):
        self.content = content
        self.confidence = confidence
        self.success_used = 0
        self.success_unused = 0
        self.failure_used = 0
        self.failure_unused = 0
        self.source_questions = {question}   # rule来源于哪些question

    def read_rule(self, rule: Dict):
        if 'content' not in rule:
            raise KeyError("The content is not in the dict")

        self.content = rule['content']
        self.confidence = rule.get('confidence', 0.0)
        self.success_used = rule.get('success_used', 0)
        self.success_unused = rule.get('success_unused', 0)
        self.failure_used = rule.get('failure_used', 0)
        self.failure_unused = rule.get('failure_unused', 0)
        self.source_questions = rule.get('source_questions', set())

        return self


class RuleBase:
    def __init__(self):
        # {rule_name: rule_instance}
        self._rule_name_2_rule_instance: Dict[str, Rule] = dict()
        self._question_2_rule = dict()

    def sample_rules(self, rule_num=100, stages=None, proportion=None, do_not_use_question: str = None,
                     override_rules: List[Rule] = None) -> List[Rule]:
        """
        按照比例采样
        同时加入不取某个question产生的规则的功能
        """
        stages = [.1, .3, 1] if not stages else stages
        proportion = [.3, .5, .2] if not proportion else proportion

        assert len(stages) == len(proportion)
        stages.insert(0, 0)

        prepared_rules = override_rules if override_rules else self._rule_name_2_rule_instance.values()

        sorted_rule_instance = [r for r in prepared_rules
                                if do_not_use_question not in r.source_questions]

        sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)
        l = len(sorted_rule_instance)

        sampled_rule_instance = []
        for i in range(len(stages)-1):
            sub_rule_instance = sorted_rule_instance[int(stages[i] * l):int(stages[i+1] * l)]
            sampled_num = min(int(rule_num * proportion[i]), len(sub_rule_instance))
            sampled_rule_instance += random.sample(sub_rule_instance, sampled_num)

        sampled_rule_instance.sort(key=operator.attrgetter('content'))
        assert len(sampled_rule_instance) <= rule_num

        return sampled_rule_instance

    # 规则采样写得不好会导致不能抽取合理的规则
    # def sample_rules(self, rule_num=100, do_not_use_question: str = None):
    #     """
    #     均匀取样，如果>=20次，若success_used : failure_used > 1，被加强采样(前50%)；否则不采样
    #     如果<20次，被均匀采样（后50%）
    #     """
    #     all_instances = list(self._rule_name_2_rule_instance.values())
    #     out_rule_num = min(rule_num, len(all_instances))
    #     if do_not_use_question is not None:     # 不取某个question产生的规则
    #         for r in self.get_q2r(do_not_use_question):
    #             ri = self._rule_name_2_rule_instance[r]
    #             assert ri in all_instances
    #             all_instances.remove(ri)
    #
    #     newcomer = []
    #     trusted = []
    #     for instance in all_instances:
    #         if instance.success_used + instance.failure_used < 20:
    #             newcomer.append(instance)
    #         else:
    #             if instance.failure_used == 0 or instance.success_used / instance.failure_used > 1:
    #                 trusted.append(instance)
    #
    #     first_half = random.sample(trusted, min(len(trusted), out_rule_num // 2))
    #     second_half = random.sample(newcomer, out_rule_num - len(first_half))
    #     sampled_rule_instance = first_half + second_half
    #     random.shuffle(sampled_rule_instance)
    #
    #     sampled_rule_instance.sort(key=operator.attrgetter('content'))
    #     assert len(sampled_rule_instance) <= out_rule_num
    #
    #     return sampled_rule_instance

    def _find_rule_class(self, added_rules: str, rules: List[str], question: 'Example') \
            -> Tuple[Set[Rule], Set[Rule]]:
        rules = [r.strip() for r in rules if r.strip() != '']
        added_rules = [r.strip() for r in added_rules.split('\n')[1:] if r.strip() != '' and r.strip() != 'Rule']
        given_rules = set([self._rule_name_2_rule_instance[r] if r in self._rule_name_2_rule_instance
                           else self._add_rules(r, question.question) for r in added_rules])
        # 第一行是prompt，第二行开始rule
        extracted_rules = set([self._rule_name_2_rule_instance[r] if r in self._rule_name_2_rule_instance
                               else self._add_rules(r, question.question) for r in rules])

        return given_rules, extracted_rules

    def _update_rule_score(self, given_rules: Set[Rule], extracted_rules: Set[Rule], question: 'Example',
                           score: float) -> float:
        for rule in given_rules & extracted_rules:
            rule.confidence += 0.1
        for rule in given_rules - extracted_rules:
            rule.confidence -= 0.001

        for rule in extracted_rules:
            rule.confidence += score

        for rule in given_rules & extracted_rules:
            rule.source_questions.add(question.question)

        for rule in extracted_rules:
            if score > 0:
                rule.success_used += 1
                print("success_used:", rule.success_used)
            else:
                rule.failure_used += 1
        for rule in given_rules - extracted_rules:
            if score > 0:
                rule.success_unused += 1
            else:
                rule.failure_unused += 1

        return score

    def update_rule(self, added_rules: str, rules: List[str], question: 'Example',
                    score: float) -> List[Rule]:
        """
        需要字符串匹配，找到就返回，找不到就创建+返回
        :param added_rules: 答题时从rulebase中抽取的规则
        :param rules: 答题时从rationale中抽取的规则
        :param question: 问题
        """
        given_rules, extracted_rules = self._find_rule_class(added_rules, rules, question)

        self._update_rule_score(given_rules, extracted_rules, question)

        return list(extracted_rules)

    def __add_rules(self, rule: str, question: str, score: float) -> Rule:
        rule_instance = Rule(content=rule, question=question, confidence=score)
        self._rule_name_2_rule_instance[rule] = rule_instance

        return rule_instance

    def _add_rules(self, rules: Union[List[str], str],
                   questions: Union[List[str], str],
                   scores: Union[List[float], float] = None) -> Union[Rule, List[Rule]]:
        """
        这里需要一个添加rule的函数，包括将字符串转为str+查重+添加
        """
        if not scores:
            scores = 1.0 if isinstance(rules, str) else [1.0] * len(rules)

        if isinstance(rules, str):
            return self.__add_rules(rules, questions, scores)
        elif isinstance(rules, list):
            new_rule_instances = [self.__add_rules(rule, question, score) for
                                  rule, question, score in zip(rules, questions, scores)]
            return new_rule_instances

    def __len__(self):
        return len(self._rule_name_2_rule_instance)

    def read_rules(self, rules: Union[List[Dict], List[str]]):
        """
        读入的是完整的rules，{'rule': "Guillermina is Christopher's daughter.", 'confidence': -22.23923742923761, 'success_used': 0, 'success_unused': 97, 'failure_used': 17, 'failure_unused': 674}
        """
        for rule_dict in rules:
            if isinstance(rule_dict, str):
                rule_dict = eval(rule_dict)

            rule = Rule(content="", question="").read_rule(rule_dict)
            self._rule_name_2_rule_instance[rule.content] = rule    # 这里应该有一个存在就不读入了的函数。
            # 或者说这里本就应该调取update来完成存储。不过暂时先这样，因为目前的update还不够灵活

    def save(self, save_path: str):
        with open(save_path, 'w') as f:
            out = [r.__dict__ for r in self._rule_name_2_rule_instance.values()]
            f.write('\n'.join([str(o) for o in sorted(out, key=lambda x: x['confidence'])]))


class DisjointSetRuleBase(RuleBase):
    def __init__(self):
        super(DisjointSetRuleBase, self).__init__()
        model_name = "all-MiniLM-L6-v2"
        self.encoder = SentenceTransformer(model_name)
        self.rule_embedding = {} # {rule class: embedding}
        self.father = {} # {rule: rule}
        self.rank = {} # {rule: rank}
        self.adjacency_matrix = defaultdict(lambda: defaultdict(lambda: 0)) # {rule: {rule: 1/0}}

    def find(self, rule: Rule):
        assert isinstance(rule, Rule)

        self.father[rule] = self.father.get(rule, rule)
        self.rank[rule] = self.rank.get(rule, 0)

        if self.father[rule] is not rule:
            self.father[rule] = self.find(self.father[rule])

        return self.father[rule]

    def union(self, rule_to_combine_1: Rule, rule_to_combine_2: Rule):
        rule_to_combine_1_root = self.find(rule_to_combine_1)
        rule_to_combine_2_root = self.find(rule_to_combine_2)

        if rule_to_combine_1_root is not rule_to_combine_2_root:
            if self.rank[rule_to_combine_1_root] > self.rank[rule_to_combine_2_root]:
                self.father[rule_to_combine_2_root] = rule_to_combine_1_root
            elif self.rank[rule_to_combine_1_root] < self.rank[rule_to_combine_2_root]:
                self.father[rule_to_combine_1_root] = rule_to_combine_2_root
            else:
                self.father[rule_to_combine_2_root] = rule_to_combine_1_root
                self.rank[rule_to_combine_1_root] += 1

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_rule_embedding(self, rule: Rule):
        if rule in self.rule_embedding:
            return self.rule_embedding[rule]

        rule_embedding = self.encoder.encode(rule.content)
        self.rule_embedding[rule] = rule_embedding

        return rule_embedding

    def create_edge(self, rule_1, rule_2):
        threshold = 0.9
        rule_1_embedding = self.get_rule_embedding(rule_1)
        rule_2_embedding = self.get_rule_embedding(rule_2)
        if (self.cosine_similarity(rule_1_embedding, rule_2_embedding) >= threshold):
            self.adjacency_matrix[rule_1][rule_2] = 1
            self.adjacency_matrix[rule_2][rule_1] = 1

            self.union(rule_1, rule_2)

    def find_cluster(self):
        """
        基于并查集拆成cluster
        """
        clusters = defaultdict(list)
        for rule in list(self._rule_name_2_rule_instance.values()): # 并行不同步的妥协，用个副本[:]
            clusters[self.find(rule)].append(rule)

        return clusters

    def average_rule_confidence(self):
        clusters = self.find_cluster()

        for cluster in clusters:
            confidence = 0
            success_used = 0
            success_unused = 0
            failure_used = 0
            failure_unused = 0
            source_questions = set()

            for rule in clusters[cluster]:
                confidence += rule.confidence
                success_used += rule.success_used
                success_unused += rule.success_unused
                failure_used += rule.failure_used
                failure_unused += rule.failure_unused
                source_questions = source_questions.union(rule.source_questions)

            confidence /= len(clusters[cluster])
            success_unused /= len(clusters[cluster])
            success_used /= len(clusters[cluster])
            failure_unused /= len(clusters[cluster])
            failure_used /= len(clusters[cluster])

            for rule in clusters[cluster]:
                rule.confidence = confidence
                rule.success_used = success_used
                rule.success_unused = success_unused
                rule.failure_used = failure_used
                rule.failure_unused = failure_unused
                rule.source_questions = source_questions

    def update_rule(self, added_rules: str, rules: List[str], question: 'Example',
                    score: float) -> List[Rule]:
        clusters = self.find_cluster() # 把这个放在create前面，为了减少并行带来的风险

        given_rules, extracted_rules = self._find_rule_class(added_rules, rules, question)

        for rule_1 in extracted_rules:
            for rule_2 in list(self._rule_name_2_rule_instance.values()) + list(extracted_rules): # 不同步妥协
                if rule_1 is not rule_2:
                    self.create_edge(rule_1, rule_2)

        # given_rules扩充为对应cluster整个
        new_given_rules = set()
        for rule in given_rules:
            new_given_rules = new_given_rules.union(clusters[self.find(rule)])

        self._update_rule_score(new_given_rules, extracted_rules, question, score)

        return list(extracted_rules)

    def sample_rules(self, rule_num=100, stages=None, proportion=None, do_not_use_question: str = None,
                     override_rules: List[Rule] = None) -> List[Rule]:

        clusters = self.find_cluster()
        override_rules = [random.choice(clusters[c]) for c in clusters] if not override_rules else override_rules

        return super().sample_rules(rule_num, stages, proportion, do_not_use_question, override_rules)


class Rationale:    # 修正到只有两个属性
    """
    top-N的结果，rationale和prediction
    """

    def __init__(self, rationale: str, prediction: str):
        self.rationale = rationale.strip()
        self.prediction = self.clean_prediction(prediction)
        self.rules = set()

    @RuleExtractionNameSpace.register("Example")
    def extract_rules_cold_start(self) -> Union[Set[str], List[str]]:
        """
        从dataset中抽取出rules
        """
        pass

    def extract_rules_training(self) -> List[str]:
        """
        从dataset中抽取出rules，目前存在的问题是未区分retrieved和new
        """
        total_rules = []
        rule_pattern = re.compile(r"<retrieved_rule>(.+?)<retrieved_rule>")
        rules = rule_pattern.findall(self.rationale)
        total_rules += [r.strip() for r in rules]
        rule_pattern = re.compile(r"<new_rule>(.+?)<new_rule>")
        rules = rule_pattern.findall(self.rationale)
        total_rules += [r.strip() for r in rules]
        self.rules = self.rules.union(set(total_rules))

        return total_rules

    @classmethod
    @PredictionCleanNameSpace.register("Example")
    def clean_prediction(self, prediction: str) -> str:
        return prediction

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

    def __repr__(self):
        return str({"rationale": self.rationale, "prediction": self.prediction})

class Example:
    def __init__(self, question: str, gold_label: str, rationale: List[Rationale] = None, *args, **kwargs):
        self.question = question.strip()
        self.gold_label = gold_label.strip()
        self.rationale = [] if rationale is None else rationale

    def update_rationale(self, rationale: Union[Dict[str, str], Rationale, List]):
        new_rationale_instance = None

        if isinstance(rationale, dict):
            new_rationale_instance = Rationale(rationale=rationale['rationale'], prediction=rationale['prediction'])
        elif isinstance(rationale, Rationale):
            new_rationale_instance = rationale
        elif isinstance(rationale, list):
            for r in rationale:
                self.update_rationale(r)
            return

        self.rationale.append(new_rationale_instance)

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
            elif isinstance(other_attr_value, str) or isinstance(other_attr_value, Rationale):
                getattr(self, attr).append(other_attr_value)
            else:
                raise TypeError("Incorrect type.")

        elif isinstance(getattr(self, attr), str):
            if isinstance(other_attr_value, str):
                 setattr(self, attr, other_attr_value)
            else:
                raise TypeError("Incorrect type.")

    def _check_QA(self, other_QA: Tuple[str, str]):
        if self.question and self.question.strip() != other_QA[0].strip():
            raise Warning("The question is not the same.")

        if self.gold_label and self.gold_label.strip() != other_QA[1].strip():
            raise Warning("The gold_label is not the same.")

        return True

    def adjust_merge_example_to_rationale(self, merge_example: Dict[str, str]) -> Dict[
        str, Union[str, Rationale, List[Rationale]]]:
        """
        将merge_example中的rationale部分(包括rationale和prediction两个key)转换为Rationale类
        """
        rationale = merge_example.pop('rationale', None)
        prediction = merge_example.pop('prediction', None)
        rationale_class = Rationale(rationale=rationale, prediction=prediction)

        merge_example['rationale'] = rationale_class

        return merge_example

    def update(self, example: [str, Dict[str, str], 'Example'], args=None):
        """
        根据example里面的更新self对应的值。请注意，这个函数用于更新某个样例的情况，而不建议将样例1改变为样例2（注意到
        list我们是直接append，而不是替换的）
        当待修改的question和gold_label有任一不同时，会抛出警告
        """
        if 'rationale' in example and 'prediction' in example:
            merge_example = self.adjust_merge_example_to_rationale(example)
        else:
            merge_example = example

        if isinstance(example, str): #parse出来一个dict
            self.parse_response(example, args)
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
        return {'question': self.question, 'gold_label': self.gold_label,
                'rationale': [r.rationale for r in self.rationale],
                'prediction': [r.prediction for r in self.rationale]}

    def parse_response(self, response: str, args=None) -> Dict[str, str]:
        """
        这里只会传入A：后面生成的部分
        """
        question_name = self.question.split('\n')[-1].strip()[10:].strip()[:-6].lower()
        pred_trigger = args.pred_trigger.lower() if args else "the answer is"

        if pred_trigger in response.lower():
            response_lst = response.lower().split(pred_trigger)
            length = len(response)
            prediction = response[length - len(response_lst[-1]):].strip()
            rationale = response[:-(len(response_lst[-1]) + len(pred_trigger))].strip()
        elif question_name in response.lower():
            response_lst = response.lower().split(question_name)
            length = len(response)
            prediction = response[length - len(response_lst[-1]):].strip()
            rationale = response[:-(len(response_lst[-1]) + len(question_name))].strip()
        else:
            rationale = response.strip()
            prediction = response.strip()

        return {'question': self.question, 'gold_label': self.gold_label,
                'rationale': rationale, 'prediction': prediction}

    def Top_k_rationale(self, k: int = 1):
        """
        返回score排名 top-k的rationale
        """
        return choices(self.rationale, k=k)

    def __repr__(self):
        return str({"question": self.question, "gold_label": self.gold_label, "rationale": self.rationale.__repr__()})

    def __hash__(self):
        return hash((self.question, self.gold_label))


class DatasetLoader:  # 命名上和torch的多加了个set
    def __init__(self, data: List[Example]):
        # {question, gold_label: data_instance}
        self._question_label_2_data_instance = dict()
        self._data_instance_list = []

        for e in data:
            self._data_instance_list.append(e)

        self._build_index()

    def __len__(self):
        return len(self._data_instance_list)

    def __getitem__(self, item):
        return self._data_instance_list[item]

    def __repr__(self):
        return " ".join([str(self._question_label_2_data_instance[d]) for d in self._question_label_2_data_instance])

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self._data_instance_list):
            raise StopIteration()
        return self._data_instance_list[self._iter_index]

    def _build_index(self):
        for data_instance in self._data_instance_list:
            question, gold_label = data_instance.question, data_instance.gold_label
            key = (question, gold_label)
            if key in self._question_label_2_data_instance:
                raise KeyError("The question and gold_label is already in the DatasetLoader")
            else:
                self._question_label_2_data_instance[key] = data_instance

    @overload
    def find(self, key: Tuple[str, str]) -> Optional[Example]: ...
    @overload
    def find(self, key: Tuple[str, str], default) -> Optional[Example]: ...

    def find(self, key, *args):
        if key in self._question_label_2_data_instance:
            return self._question_label_2_data_instance[key]
        else:
            if args:
                return args[0]
            else:
                raise KeyError("The question and gold_label pair is not in the DatasetLoader")