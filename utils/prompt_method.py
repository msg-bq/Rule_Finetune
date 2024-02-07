import random
import re
from typing import Tuple

from utils.ExtraNameSpace import PromptMethodNameSpace

def add_HtT_rule_prefix_suffix(htt_rule: str, prefix: bool, suffix: bool):
    """
    A's B is C -> <A><B>A's B is C</B></A>
    """
    htt_rule = remove_HtT_rule_prefix_suffix(htt_rule)
    pattern = r"(\w+)'s (\w+) is \w+"
    matches = re.findall(pattern, htt_rule)
    if not matches or len(matches[0]) != 2:
        pattern = r"(\w+)’s (\w+) is \w+"
        matches = re.findall(pattern, htt_rule)
        if not matches or len(matches[0]) != 2:
            return htt_rule
    A, B = matches[0]

    out = ''
    if prefix:
        out += f"<{A}><{B}>"
    out += f"{htt_rule}"
    if suffix:
        out += f"</{B}></{A}>"
    return out


def remove_HtT_rule_prefix_suffix(htt_rule):
    useless_pattern = re.compile(r"(<.*?>)")
    for p in useless_pattern.findall(htt_rule):
        htt_rule = htt_rule.replace(p, '')
    assert '<' not in htt_rule
    return htt_rule.strip()
    
    
@PromptMethodNameSpace.register("Default")
def prompt_demos(args, demos: str, added_rules: str) -> Tuple[str, str]:
    assert args.train | args.test
    demos = demos.replace("<retrieved_rule>", "<rule>")
    demos = demos.replace("<new_rule>", "<rule>")
    demos = demos.replace("<Begin>", "<rule>").replace("</Begin>", "<rule>").replace("</End>", "<rule>").replace("<End>", "<rule>")


    rule_pattern = re.compile(r"<rule>(.+?)<rule>")
    rules = rule_pattern.findall(demos)

    cnt = 0
    for rule in rules:
        if rule.strip() in added_rules:
            cnt += 1

    if cnt < 5:  # 这个阈值可以考虑随epoch增大
        chosen_rules = random.sample(rules, min(len(rules), 5 - cnt))
        added_rules = added_rules.strip() + "\n" + "\n".join(chosen_rules) + "\n\n"

    for rule in rules: # rationale(旧)， sampled_rules(新)。 retrieve和new要变的
        if rule.strip() in added_rules:
            demos = demos.replace("<rule>" + rule + "<rule>", "<retrieved_rule> " + rule.strip() + " <retrieved_rule>")
        else:
            demos = demos.replace("<rule>" + rule + "<rule>", "<new_rule> " + rule.strip() + " <new_rule>")

    out = ''
    out += demos
    return out, added_rules

@PromptMethodNameSpace.register("HtT_version")
def prompt_demos(args, demos: str, added_rules: str) -> Tuple[str, str]:
    assert args.train | args.test
    rule_pattern1 = re.compile(r"we have(.+?). So the")
    rule_pattern2 = re.compile(r"we retrieve(.+?). So the")
    rules = rule_pattern1.findall(demos) + rule_pattern2.findall(demos)

    cnt = 0
    for rule in rules:
        if rule.strip() in added_rules:
            cnt += 1

    if cnt < 5:  # 这个阈值可以考虑随epoch增大
        chosen_rules = random.sample(rules, min(len(rules), 5 - cnt))
        added_rules = added_rules.strip() + "\n" + "\n".join(chosen_rules) + "\n\n"

    for rule in rules:  # rationale(旧)， sampled_rules(新)。 retrieve和new要变的
        if rule.strip() in added_rules:
            demos = demos.replace(f"we have{rule}", f"we retrieve {add_HtT_rule_prefix_suffix(rule, prefix=True, suffix=True)}")
        else:
            demos = demos.replace(f"we retrieve{rule}", f"we have {add_HtT_rule_prefix_suffix(rule, prefix=True, suffix=True)}")

    out = ''
    out += demos
    return out, added_rules