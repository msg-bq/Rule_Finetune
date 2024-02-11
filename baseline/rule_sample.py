import operator

from utils.data import RuleBase, DisjointSetRuleBase


def sample_rule_confidence(rule_base: RuleBase, max_seq_len=500):
    if isinstance(rule_base, DisjointSetRuleBase):
        clusters = rule_base.find_cluster()
        sorted_rule_instance = [clusters[cluster][0] for cluster in clusters]
        sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)
    else:
        sorted_rule_instance = list (rule_base._rule_name_2_rule_instance.values())
        sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)

    l = len(sorted_rule_instance)
    sampled_rule_instance = []
    for rule in sorted_rule_instance:
        sampled_rule_instance.append(rule)

        max_seq_len -= len(rule.content.split())
        if max_seq_len < 0:
            break

    # sampled_rule_instance.sort(key=operator.attrgetter('content'))

    return sampled_rule_instance


def sample_rule_confidence_with_high_low_mark(rule_base: RuleBase, max_seq_len=500):
    if isinstance(rule_base, DisjointSetRuleBase):
        clusters = rule_base.find_cluster()
        sorted_rule_instance = [clusters[cluster][0] for cluster in clusters]
        sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)
    else:
        sorted_rule_instance = list (rule_base._rule_name_2_rule_instance.values())
        sorted_rule_instance.sort(key=operator.attrgetter('confidence'), reverse=True)

    l = len(sorted_rule_instance)
    sampled_rule_instance_high_confidence = []
    sampled_rule_instance_low_confidence = []
    for i in range(l):
        if sorted_rule_instance[i].confidence >= 1:
            sampled_rule_instance_high_confidence.append(sorted_rule_instance[i])
            max_seq_len -= len(sorted_rule_instance[i].content.split())

        if sorted_rule_instance[l-i-1].confidence < 1:
            sampled_rule_instance_low_confidence.append(sorted_rule_instance[l-i-1])
            max_seq_len -= len(sorted_rule_instance[l-i-1].content.split())

        if max_seq_len < 0:
            break

    # for rule in sorted_rule_instance:
    #     sampled_rule_instance_high_confidence.append(rule)
    #
    #     max_seq_len -= len(rule.content.split())
    #     if max_seq_len < 0:
    #         break

    for rule in sorted_rule_instance:
        if rule.confidence >= 1 and not rule.content.startswith("✓"):
            rule.content = "✓ " + rule.content
        elif rule.confidence < 1 and not rule.content.startswith("✗"):
            rule.content = "✗ " + rule.content

    sampled_rule_instance = sampled_rule_instance_high_confidence + sampled_rule_instance_low_confidence
    # for rule in sampled_rule_instance_high_confidence:
    #     if not rule.content.endswith(", ✓"):
    #         rule.content += ", ✓"
    #
    # for rule in sampled_rule_instance_low_confidence:
    #     if not rule.content.endswith(", ✗"):
    #         rule.content += ", ✗"

    # sampled_rule_instance.sort(key=operator.attrgetter('content'))

    return sampled_rule_instance

def sample_rule():
    pass

sample_rule_strategy = {'confidence': sample_rule_confidence,
                        'confidence_mark': sample_rule_confidence_with_high_low_mark}