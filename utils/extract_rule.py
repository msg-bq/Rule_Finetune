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


