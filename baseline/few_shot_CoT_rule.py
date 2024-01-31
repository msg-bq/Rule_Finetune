import argparse
import operator
from concurrent.futures import ThreadPoolExecutor

from RuleFinetune.RuleTrainer import Trainer
from RuleFinetune.autoCoT import prompt_rules
from utils.data import Rationale, Example, RuleBase
from utils.llm_models.call_openai import call_openai
from utils.read_datasets import read_datasets

demos = """Context: Donald wanted to go visit his grandfather Jason. His dad Michael said okay and drove him to over to Jason house.
Question: Jason is Michael's what?
Answer: <retrieved_rule> Children's grandparents are their parents' parents <retrieved_rule> Given the context, Donald is visiting his grandfather Jason. Michael, Donald's father, is driving him there. According to the rule, if Jason is Donald's grandfather, then Jason must be Michael's father. Therefore, Jason is Michael's father. The answer is father.

Context: Michael is taking his son Donald out for coffee. Donald invited his uncle Christopher to dinner
Question: Christopher is Michael's what?
Answer: <new_rule> Brothers' children are your nieces or nephews. <new_rule> <new_rule> Your child's uncle is either your brother or your spouse's brother. <new_rule> Given the context: Michael is taking his son Donald out for coffee, and Donald invited his uncle Christopher to dinner. Applying the rules: Since Christopher is Donald's uncle, we apply the second rule: Your child's uncle is either your brother or your spouse's brother. Since the relationship is through Michael's son, we can infer that Christopher is Michael's brother. Therefore, Christopher is Michael's brother. The answer is brother.

Context: Christopher went out for pizza with his daughter Lucille and his brother Dwight.
Question: Lucille is Dwight's what?
Answer: <retrieved_rule> Children of siblings are cousins to each other. <retrieved_rule> <retrieved_rule> Siblings are individuals who share at least one parent. <retrieved_rule> <retrieved_rule> The child of one's sibling is one's niece or nephew. <retrieved_rule> Given the context: 1. Christopher is the father of Lucille. 2. Dwight is the brother of Christopher. Applying the rules: - Since Dwight is Christopher's brother, they are siblings (Rule 2). - Lucille, being the daughter of Christopher, is the child of Dwight's sibling (Christopher). - Therefore, Lucille is Dwight's niece (Rule 3). The answer is niece.

Context: Michael and his son Donald went to the store to by bread. Donald took his brother Russell out to get drinks after a long work week.
Question: Russell is Michael's what?
Answer: <retrieved_rule> Children of the same parents are siblings to each other. <retrieved_rule> <new_rule> If Person A is the parent of Person B, and Person B is a sibling of Person C, then Person A is the parent of Person C. <new_rule> From the context: 1. Donald is Michael's son. 2. Donald took his brother Russell out. Applying the rules: - Since Donald is Michael's son, and Russell is Donald's brother, we apply the first rule and acknowledge that Donald and Russell are siblings. - Applying the second rule, since Michael is the parent of Donald, and Donald is a sibling of Russell, it follows that Michael is also the parent of Russell. Answer: Russell is Michael's son. The answer is son.

Context: Jason went bike riding his grandson Joe. Joe's father Dwight was skateboarding.
Question: Jason is Dwight's what?
Answer: <retrieved_rule> Parents of one's parents are called grandparents. <retrieved_rule> <new_rule> Fathers are male parents. <new_rule> Since Joe is Jason's grandson, Jason must be the grandparent of Joe. Given that Dwight is Joe's father and therefore a male parent, we can deduce that Jason is Dwight's father or mother. Since the question asks specifically about Jason, who is involved in an activity with his grandson, it is most likely that Jason is male (although the information provided does not explicitly state Jason's gender). Assuming Jason is male: <new_rule> Male grandparents are called grandfathers. <new_rule> Therefore, Jason is Dwight's father. To summarize: Jason (grandfather of Joe) --> Dwight's father. The answer is father."""


parser = argparse.ArgumentParser(description="Rule-Finetune")
args = parser.parse_args()
args.data_dir = "../data/CLUTRR"
args.dataset = "CLUTRR"

train_dataset, valid_dataset, test_dataset = read_datasets(args)

rule_base = RuleBase()
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

def eval_step(example: Example):
    global rule_base
    added_rules = prompt_rules(sample_rule(rule_base))
    demo_prompt = '''Next, I will give you some examples to indicate how to use existed rules by <retrieved_rule> ''' \
                  '''<retrieved_rule>tag and add the new rules into knowledge base by <new_rule> <new_rule> tag.
                  Examples: \n'''

    # prompt = added_rules + '\n' + 'Then, please answer this question: \n' + example.question + "\nAnswer: Let's think step by step."
#     rules ='\n'.join([
#         # str(idx+1)+':\t'+rn.content
#         rn.content
#         for idx, rn in enumerate(sample_rule(rule_base))])
#     prompt = """Instruction: When you answer the questions, try to use the provided knowledge whenever possible. Try
# not to invent knowledge by yourself unless necessary.
# Knowledge:
# {}\n
# """.format(rules) \
#     + demos.replace('<retrieved_rule>', '<rule>').replace('<new_rule>', '<rule>') + '\n' + \
#     example.question + "\nAnswer:"
    prompt = demos.replace('<retrieved_rule>', '<rule>').replace('<new_rule>', '<rule>') + '\n' + example.question + "\nAnswer:"
        #          demo_prompt + demos + '\n' + \
        #          'Then, please answer this question: \n' + \
        # """Task: A benchmark dataset generator to test relational reasoning on text. Instruction: Following are several existed rules in knowledge base. When you answer the questions, """ \
        #   """try to use the provided rules whenever possible in <retrieved_rule>xxx<retrieved_rule> format. """ \
        #   """Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to """ \
        #   """establish your own rules in <new_rule>xxx<new_rule> format.""" \
        #   """Example of the task:
# {}
#
# Strategy:
# 1. Identify Context and Question: Extract the context (path of relations) and the question (relation to identify) from the task description.
# 2. Parse the Context: Break down the context into individual relations. This involves separating each relation (e.g., son, sister, uncle) and listing them in the order they appear.
# 3. Apply Existing Rules: Use the provided rules in the knowledge base to interpret the relations. This means applying the <retrieved_rule> format to understand how each relation affects the familial connection.
# 4. Establish New Rules if Necessary: If the existing rules are insufficient to determine the relation, establish new rules. This is done by logically deducing relations that are not covered by existing rules, using the <new_rule> format.
# 5. Trace the Path of Relations: Starting from the initial person, trace the path of relations one by one. Apply each relation to understand how it changes the familial connection.
# 6. Answer the Question: After applying all relations in the path, determine the final relation between the initial and the final person in the context. Answer the question using this relation.
# 7. Verify the Answer: Double-check the answer by reviewing the path of relations and ensuring that all rules (existing or new) have been correctly applied.
# 8. Adjust if Necessary: If the answer seems incorrect, revisit the steps, especially the application of rules, and make necessary adjustments.
#
# The strategy consists of a sequence of subtasks for solving the task. Please execute the strategy on the provided example. For executing, you need to write a step-by-step solution to the example based on the subtasks. The solution must satisfy the following requirements:
# - Adjust and execute these subtasks for this example.
# - Compute as many intermediate results as possible.
# - The answer obtained from the solution must be the same as the original answer.
# The result must be in the following format:
# Question: Question in the provided example
# Solution: Solution obtained based on the subtasks in the strategy
# Answer: Answer in the provided example, which must include the string ‘The answer is ’""".format(example.question.strip())

#              example.question.strip() + "\nAnswer:" + \
#     """\nStrategy:
# 1. Identify Context and Question: Extract the context (path of relations) and the question (relation to identify) from the task description.
# 2. Parse the Context: Break down the context into individual relations. This involves separating each relation (e.g., son, sister, uncle) and listing them in the order they appear.
# 3. Apply Existing Rules: Use the provided rules in the knowledge base to interpret the relations. This means applying the <retrieved_rule> format to understand how each relation affects the familial connection.
# 4. Establish New Rules if Necessary: If the existing rules are insufficient to determine the relation, establish new rules. This is done by logically deducing relations that are not covered by existing rules, using the <new_rule> format.
# 5. Trace the Path of Relations: Starting from the initial person, trace the path of relations one by one. Apply each relation to understand how it changes the familial connection.
# 6. Answer the Question: After applying all relations in the path, determine the final relation between the initial and the final person in the context. Answer the question using this relation.
# 7. Verify the Answer: Double-check the answer by reviewing the path of relations and ensuring that all rules (existing or new) have been correctly applied.
# 8. Adjust if Necessary: If the answer seems incorrect, revisit the steps, especially the application of rules, and make necessary adjustments."""

    response = call_openai(prompt)
    rationale = example.parse_response(response)
    prediction = Rationale.clean_prediction(rationale['prediction'])
    print(rationale, prediction, example.gold_label)
    return prediction, example.gold_label

correct_cnt = 0
final_dataset = train_dataset
with ThreadPoolExecutor(max_workers=200) as executor:
    futures = [executor.submit(eval_step, example) for example in final_dataset]
    for future in futures:
        prediction, gold_label = future.result()
        if prediction == gold_label:
            correct_cnt += 1

print(f"测试集上的准确率为：{correct_cnt / len(final_dataset)}")