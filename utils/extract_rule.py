import re
from typing import Union, Set, List

from utils.ExtraNameSpace import RuleExtractionNameSpace


@RuleExtractionNameSpace.register("default1")
def extract_rules_cold_start(self) -> Union[Set[str], List[str]]:
    if self.rules:
        return self.rules

    rule_pattern = re.compile(r"<Begin>(.+?)</End>")
    rules = rule_pattern.findall(self.rationale)
    rules = [r.strip() for r in rules if len(r.split()) > 2 and r.strip() != '']
    self.rules = set(rules)

    return rules

@RuleExtractionNameSpace.register("default2")
def extract_rules_cold_start(self) -> Union[Set[str], List[str]]:
    if self.rules:
        return self.rules

    rule_pattern = re.compile(r"<Begin>(.+?)</End>")
    rules = rule_pattern.findall(self.rationale)
    rules = [r.strip() for r in rules if len(r.split()) > 2 and r.strip() != '']
    self.rules = set(rules)

    return rules

@RuleExtractionNameSpace.register("default3")
def extract_rules_cold_start(self) -> Union[Set[str], List[str]]:
    if self.rules:
        return self.rules

    rule_pattern = re.compile(r"<Begin>(.+?)</End>")
    rules = rule_pattern.findall(self.rationale)
    rules = [r.strip() for r in rules if len(r.split()) > 2 and r.strip() != '']
    self.rules = set(rules)

    return rules

@RuleExtractionNameSpace.register("HtT_version")
def extract_rules_cold_start(self) -> Union[Set[str], List[str]]:
    if self.rules:
        return self.rules

    rule_pattern = re.compile(r"we have(.+?). So the")
    rules = rule_pattern.findall(self.rationale)
    rules = [r.strip() for r in rules if len(r.split()) > 2 and r.strip() != '']
    self.rules = set(rules)

    return rules

@RuleExtractionNameSpace.register("Default")
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

@RuleExtractionNameSpace.register("HtT_version")
def extract_rules_training(self) -> List[str]:
    """
    从dataset中抽取出rules，目前存在的问题是未区分retrieved和new
    """
    total_rules = []
    rule_pattern1 = re.compile(r"we have(.+?). So the")
    rule_pattern2 = re.compile(r"we retrieve(.+?). So the")
    rules1 = rule_pattern1.findall(self.rationale)
    rules2 = rule_pattern2.findall(self.rationale)

    total_rules += [r.strip() for r in rules1]
    total_rules += [r.strip() for r in rules2]

    self.rules = self.rules.union(set(total_rules))

    return total_rules