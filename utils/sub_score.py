"""
做一些细致的打分
"""
import operator
import re
import sys
from collections import defaultdict
from typing import List, Union

from utils.data import RuleBase
from utils.kinship_calculator import kenship_calculator

rule_base = RuleBase()
used_rules_cnt = defaultdict(lambda: 0)

with open(f"../experiment/rule_base_final", encoding="utf8") as f:
    rules = [l for l in f.readlines() if l.strip()]
    rule_base.read_rules(rules)

def sample_rule(rule_base: RuleBase):
    sorted_rule_instance = list (rule_base._rule_name_2_rule_instance.values())
    sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)
    l = len(sorted_rule_instance)

    sampled_rule_instance = []
    max_seq_len = 1800
    for rule in sorted_rule_instance:
        sampled_rule_instance.append(rule)

        max_seq_len -= len(rule.content.split())
        if max_seq_len < 0:
            break

    sampled_rule_instance.sort(key=operator.attrgetter('content'))

    return sampled_rule_instance

added_rules = " ".join(rule.content for rule in sample_rule(rule_base)).lower()

sample_rule_base = ["Aunt's sister is aunt", "Brother's aunt is aunt", "Brother's brother is brother",
                 "Brother's daughter is niece", "Brother's father is father", "Brother's grandfather is grandfather",
                 "Brother's grandmother is grandmother", "Brother's mother is mother", "Brother's sister is sister",
                 "Brother's son is nephew", "Brother's uncle is uncle", "Brother's wife is sister-in-law",
                 "Brother-in-law's daughter is niece", "Brother-in-law's father is father-in-law",
                 "Brother-in-law's mother is mother-in-law", "Brother-in-law's son is nephew",
                 "Daughter's aunt is sister", "Daughter's brother is son", "Daughter's daughter is granddaughter",
                 "Daughter's grandfather is father", "Daughter's grandmother is mother", "Daughter's husband is son-in-law",
                 "Daughter's sister is daughter", "Daughter'sson is grandson", "Daughter's uncle is brother",
                 "Daughter-in-law's daughter is granddaughter", "Daughter-inlaw's son is grandson", "Father's brother is uncle",
                 "Father's daughter is sister", "Father's father is grandfather", "Father's mother is grandmother",
                 "Father's sister is aunt", "Father's son is brother", "Father's wife is mother", "Granddaughter's brother is grandson",
                 "Granddaughter's father is son", "Granddaughter's mother is daughter", "Granddaughter's sister is granddaughter",
                 "Granddaughter's uncle is son", "Grandfather's daughter isaunt", "Grandfather's son is uncle",
                 "Grandmother's daughter is aunt", "Grandmother's son is uncle", "Grandson's brother is grandson",
                 "Grandson's father is son", "Grandson's mother is daughter", "Grandson's sister isgranddaughter",
                 "Grandson's uncle is son", "Husband's daughter is daughter", "Husband's father is father-inlaw",
                 "Husband's granddaughter is granddaughter", "Husband's grandson is grandson", "Husband's mother ismother-in-law",
                 "Husband's son is son", "Mother's brother is uncle", "Mother's daughter is sister", "Mother'sfather is grandfather",
                 "Mother's mother is grandmother", "Mother's sister is aunt", "Mother's son is brother", "Nephew's grandfather is father",
                 "Nephew's grandmother is mother", "Nephew's sister is niece", "Niece'sbrother is nephew", "Niece's uncle is brother",
                 "Self's brother is brother", "Sister's brother is brother", "Sister'sdaughter is niece", "Sister's father is father",
                 "Sister's grandfather is grandfather", "Sister's grandmother is grandmother", "Sister's husband is brother-in-law",
                 "Sister's mother is mother", "Sister's sister is sister", "Sister's25Large Language Models can Learn Rulesson is nephew",
                 "Sister-in-law's daughter is niece", "Sister-in-law's father is father-in-law", "Sister-in-law's mother is mother-in-law",
                 "Sister-in-law's son is nephew", "Son's aunt is sister", "Son's brother is son", "Son's daughter is granddaughter",
                 "Son's grandfather is father", "Son's grandmother is mother", "Son's sister isdaughter", "Son's son is grandson",
                 "Son's uncle is brother", "Son's wife is daughter-in-law", "Son-in-law's son isgrandson",
                 "Step-daughter's grandmother is mother", "Uncle's sister is aunt", "Wife's brother is brother-in-law",
                 "Wife's daughter is daughter", "Wife's father is father-in-law", "Wife's granddaughter is granddaughter",
                 "Wife'sgrandson is grandson", "Wife's mother is mother-in-law", "Wife's son is son", "sister's aunt is aunt",
                 "son's father is husband", "son's mother is wife", "sister's uncle is uncle", "sister's son is nephew",
                 "Grandmother's son is father", "grandmother's daughter is mother", "grandfather's son is father",
                 "grandfather's daughter is mother", "grandson's father is son", "grandson's mother is daughter",
                 "granddaughter's father is son", "granddaughter's mother is daughter", "mother's daughter is sibling",
                 "mother's son is sibling", "father's daughter is sibling", "father's son is sibling", "brother's sister is sibling",
                 "brother's brother is sibling", "sister's brother is sibling", "sister's sister is sibling",
                 "uncle's daughter is cousin", "uncle's son is cousin", "aunt's daughter is cousin", "aunt's son is cousin",
                 "cousin's daughter is niece", "cousin's son is nephew", "sister's daughter is niece", "brother's daughter is niece",
                 "brother's son is nephew", "sister's son is nephew", "sister's brother is still brother", "cousin's father is uncle",
                 "cousin's mother is aunt", "cousin's sister is cousin", "cousin's brother is cousin",
                 "nephew's son is grand-nephew", "grand-nephew's father is nephew", "sibling's sister is sister",
                 "sibling's brother is brother", "sibling's mother is mother", "sibling's father is father",
                 "brother's sister is still sister", "sister's brother is still brother", "sister's sister is still sister",
                 "brother's brother is still brother", "brother's sister is still sister", "sister's brother is still brother",
                 "sibling's sister is sibling", "grandmother's husband is grandfather", "grandfather's wife is grandmother",
                 "mother's father is grandfather", "father's mother is grandmother", "father's father is grandfather",
                 "mother's mother is grandmother", "cousin's grandmother is grandmother", "cousin's grandfather is grandfather",
"cousin's uncle is uncle", "father's wife is mother", "mother's husband is father",
                 ]

# def kenship_calculator(relation_inference_text: str):
#     """
#     只用于测试r1's r2 is r3是否正确，输入是Mother's father is grandmother这种
#     """
#     global sample_rule_base
#     rule_base = [rule.strip().lower() for rule in sample_rule_base]
#
#     relation_inference_text = relation_inference_text.strip().lower()
#
#     return relation_inference_text in rule_base

def list_startswith(lst1: list, lst2: list):
    """
    判断lst1是否以lst2开头
    """
    if len(lst1) < len(lst2):
        return False

    return lst1[:len(lst2)] == lst2

def list_endswith(lst1: list, lst2: list):
    """
    判断lst1是否以lst2结尾
    """
    if len(lst2) == 0:
        return True

    if len(lst1) < len(lst2):
        return False

    return lst1[-len(lst2):] == lst2

def parse_response(question, response, added_rules: Union[List[str], str] = []):
    """
    拆分出打分小项，包括末尾括号的数字、每一步的结论、使用规则的情况
    比如对于下面的question和response：
    Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
    Question: Anthony is Alan's what?
    Response:
    The relation path is daughter, uncle, son (3)
    The first two relations is daughter and uncle. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children", so daughter's uncle is brother, then the relations are reduced to brother, son (3 to 2).
    The first two relations is brother and son. We retrieve "The child of one's sibling is one's niece or nephew", so brother's son is nephew, then the relations are reduced to nephew (2 to 1).
    Therefore, the answer is nephew.

    返回的是：
    1. 判断context里的relation path长度，并检查response的第一行是否正确，后续是否是逐步减小
    2. 检查每行的结论是否正确，即对应的reduced to，是否数量正确，是否内容正确
    3. first two relations是否正确
    4. 选取的规则是否真的在提供的rule base里，而we have又是否是真的不在rule base里
    5. answer是否正确
    6. [optional] 对于要求输出so daughter's uncle is brother这种每一步的结论的情况，检查中间推理步骤是否正确
    """
    question = question.strip().lower()
    response = response.strip().lower()
    # added_rules = [rule.strip().lower() for rule in added_rules]

    scores = defaultdict(list)
    global used_rules_cnt

    # 1. 判断context里的relation path长度，并检查response的第一行是否正确，后续是否是逐步减小
    pattern = re.compile(r"the relations on the path from (.*) to (.*) are (.*)\.([ |\n])")
    relation_path = pattern.search(question).group(3).strip().split(",")
    relation_path = [relation.strip() for relation in relation_path]
    relation_path_length = len(relation_path)

    length_reduce_list = []
    step_relation_path_list = []
    first_two_relations_list = []

    first_line_pattern = re.compile(r"the relation path is (.*).")
    match = first_line_pattern.search(response.split("\n")[0])
    if match:
        pattern_length = re.compile(r"(.*) \((\d+)\)")
        match_length = pattern_length.search(match.group(1))

        if match_length:
            match = match_length

        step_relation_path = match.group(1).strip().split(",")
        step_relation_path = [relation.strip() for relation in step_relation_path]
        step_relation_path_list.append(step_relation_path)
        if step_relation_path != relation_path:
            print("Error: first line error")
            # return False
            scores["step_relation_path"].append(0)
        else:
            scores["step_relation_path"].append(1)

        if match_length:
            length = int(match.group(2))
            length_reduce_list.append((length, length))
            if length != relation_path_length:
                print("Error: length error")
                # return False
                scores["length_reduce"].append(0)
            else:
                scores["length_reduce"].append(1)
    else:
        step_relation_path_list.append(relation_path)


    step_cnt = 1 # 进行到了第几步
    for line in response.split("\n")[1:-1]:
        pattern = re.compile(r"the first relation pair is (.*) and (.*). we")
        match = pattern.search(line)
        if match:
            first_relation = match.group(1)
            second_relation = match.group(2)
            first_two_relations_list.append((first_relation, second_relation))

            if not list_startswith(step_relation_path_list[-1], [first_relation, second_relation]):
                print([first_relation, second_relation], step_relation_path_list[-1])
                print("Error: first two relations error")
                # return False
                scores["first_two_relations"].append(0)
            else:
                scores["first_two_relations"].append(1)

        pattern = re.compile(r"the relations are reduced to (.*).")
        match = pattern.search(line)

        if match:
            text = match.group(1)
            pattern_length_reduce = re.compile(r"(.*) \((\d+) to (\d+)\)")
            match_length_reduce = pattern_length_reduce.search(text)

            if match_length_reduce:
                match = match_length_reduce

            step_relation_path = match.group(1).strip().split(", ")
            step_relation_path_list.append(step_relation_path)

            if not list_endswith(relation_path, step_relation_path[1:]):
                print("Error: step relation path error")
                # return False
                scores["step_relation_path"].append(0)
            else:
                scores["step_relation_path"].append(1)

            if match_length_reduce:
                last_length = int(match.group(2))
                current_length = int(match.group(3))
                if last_length - current_length != 1:  # 实在上一个没生成就算了
                    print("Error: length reduce error")
                    # return False
                    scores["length_reduce"].append(0)
                else:
                    scores["length_reduce"].append(1)

                length_reduce_list.append((last_length, current_length))
    # 2. 检查每行的结论是否正确，即对应的reduced to，是否数量正确，是否内容正确
    # done

    # 3. first two relations是否正确
    # done

    # 4. 选取的规则是否真的在提供的rule base里，而we have又是否是真的不在rule base里
    # for line in response.split("\n")[1:-1]: # 手动保留值得检查的行
        pattern = re.compile(r"we retrieve \"(.*)\" from provided knowledge bas") #from provided knowledge bas
        match = pattern.search(line)

        if match:
            rule = match.group(1)
            used_rules_cnt[rule] += 1

            if rule not in added_rules:
                print("Error: rule not in added rules")
                print("没有正确引用规则：", rule)
                # return False
                scores["rule_retrieve"].append(0)
            else:
                scores["rule_retrieve"].append(1)

        # pattern = re.compile(r"we have \"(.*)\", so")
        # match = pattern.search(line)
        #
        # if match:
        #     rule = match.group(1)
        #
        #     if rule in added_rules:
        #         print("Error: rule have been in added rules")
        #         # return False
        #         scores["rule_new"].append(0)
        #     else:
        #         scores["rule_new"].append(1)

        pattern1 = re.compile(r"we retrieve \"(.*)\" from provided knowledge bas.*, so (.*) is (.*), then")
        #re.compile(r"we retrieve \"(.*)\", so (.*) is (.*), then")
        pattern2 = re.compile(r"we have \"(.*)\"  from provided knowledge bas.*, so (.*) is (.*), then")
        match1 = pattern1.search(line)
        match2 = pattern2.search(line)
        match = match1 if match1 else match2

        first_relation = ""
        result_relation = ""
        if match:
            first_relation = match.group(2)
            result_relation = match.group(3)
        else:
            if len(first_two_relations_list) >= step_cnt:
                first_relation = f"{first_two_relations_list[-1][0]}'s {first_two_relations_list[-1][1]}"
                result_relation = step_relation_path_list[-1][0]
                step_cnt += 1

        if first_relation and result_relation:
            check_fact = f"{first_relation} is {result_relation}"
            if not kenship_calculator(check_fact):
                print("Error: rule not in rule base")
                # donot_print = ["mother's son is son", "son's grandmother is grandmother", "son's brother is brother",
                #                "son's sister is sister", "brother's mother is grandmother", "son's uncle is father",
                #                "nephew's sister is sister", "aunt's sister is sister", "mother's son is nephew",
                #                "nephew's brother is uncle", "grandfather's sister is sister", "father's daughter is daughter",
                #                "daughter's brother is brother", "father's brother is brother", "daughter's brother is brother",
                #                "daughter's sister is sister", "son's aunt is aunt", "cousin's son is cousin", "cousin's brother is nephew",
                #                "daughter's mother is aunt", "cousin's brother is nephew", "nephew's uncle is uncle",
                #                "nephew's daughter is niece", "nephew's son is son", "sister's mother is daughter",
                #                "daughter's son is son", "son's mother is mother", "daughter's father is father",
                #                "father's daughter is niece", "daughter's brother is sibling", "mother's daughter is daughter",
                #                "daughter's mother is mother", "nephew's aunt is aunt", "cousin's son is great nephew",
                #                "great nephew's brother is great uncle",
                #                "father's son is son", "nephew's sister is also sister", ]
                # if check_fact not in donot_print:
                #     print("找不到的规则是：", check_fact)
                # return False
                scores["step_inference_accuracy"].append(0)
            else:
                scores["step_inference_accuracy"].append(1)

    # 5. answer是否正确
    # done

    # 6. [optional] 对于要求输出so daughter's uncle is brother这种每一步的结论的情况，检查中间推理步骤是否正确
    # done

    if len(length_reduce_list) != len(relation_path):
        print("Error: length reduce is not enough")
        # return False
        scores["length_reduce"].extend([0] * (len(relation_path) - len(length_reduce_list))) #这里就没有按照顺序补0了，而是直接补到最后了

    # scores = {key: sum(value) / len(value) for key, value in scores.items()}

    return scores

test_question = """Context: The relations on the path from Sharon to Leo are husband, son, father, son, aunt, son.\nQuestion: Leo is Sharon's what?"""
# test_response = """The relation path is daughter, uncle, son (3)
# The first two relations is daughter and uncle. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children", then the relations are reduced to father, son (3 to 2).
# The first two relations is brother and son. We retrieve "The child of one's sibling is one's niece or nephew", so brother's son is nephew, then the relations are reduced to nephew (2 to 1).
# Therefore, the answer is nephew."""

test_response = """The relation path is husband, son, father, son, aunt, son (6).
The first relation pair is husband's son, because "Sons' fathers are those sons' fathers' wives' husbands." So the relations are reduced to son, father, son, aunt, son (6 to 5).
The first relation pair is son's father, because "Sons' fathers are those sons' fathers' wives' husbands." So the relations are reduced to father, son, aunt, son (5 to 4).
The first relation pair is father's son, because "Children of any person are sons and daughters of that person's parents." So the relations are reduced to son, aunt, son (4 to 3).
The first relation pair is son's aunt, because "Siblings of a parent are aunts or uncles to their sibling's children." So the relations are reduced to aunt, son (3 to 2).
The first relation pair is aunt's son, because "The child of one's sibling is one's niece or nephew." So the relation is reduced to nephew (2 to 1).
Therefore,"""

# add_rules = ["Siblings of a parent are aunts or uncles to their sibling's children", "The child of one's sibling is one's niece or nephew"]
# #
# scores = parse_response(test_question, test_response, add_rules)
# print(scores)
#
# scores = []
# with open(r"C:\dev_D\software\wechat\files\WeChat Files\wxid_8edjcjs0zlqf22\FileStorage"
#           r"\File\2024-01\Rule_Finetune(2)(1)\Rule_Finetune\baseline\five_shot_symbolic6.txt",
#           "r", encoding='utf8') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = eval(line)
#         question = line["question"]
#         response = line["rationale"]
#         print(response)
#         score = parse_response(question, response, added_rules)
#         print(score)
#         scores.append(score)
#
# for key in ["step_relation_path", "length_reduce", "first_two_relations", "rule_retrieve", "step_inference_accuracy"]:
#     success_key_score = [score[key] for score in scores if key in score]
#     s = []
#     for i in success_key_score:
#         s.extend(i)
#
#     print(key, sum(s)/len(s))
#     # if len(success_key_score):
#         # print(key, sum(success_key_score)/len(success_key_score))
#
#
# sorted_keys = sorted(used_rules_cnt.items(), key=lambda x: x[1])
# for k, v in sorted_keys:
#     print(k, v)