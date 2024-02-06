import random
import re
from typing import Tuple

from utils.ExtraNameSpace import PromptMethodNameSpace

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
    pass