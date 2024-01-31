import operator

from utils.data import RuleBase


def sample_rule_confidence(rule_base: RuleBase, max_seq_len=500):
    sorted_rule_instance = list (rule_base._rule_name_2_rule_instance.values())
    sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)
    l = len(sorted_rule_instance)

    sampled_rule_instance = []
    for rule in sorted_rule_instance:
        sampled_rule_instance.append(rule)

        max_seq_len -= len(rule.content.split())
        if max_seq_len < 0:
            break

    sampled_rule_instance.sort(key=operator.attrgetter('content'))

    return sampled_rule_instance

def sample_rule():
    pass

sample_rule_strategy = {'confidence': sample_rule_confidence}