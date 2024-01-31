import argparse
import operator
from concurrent.futures import ThreadPoolExecutor
from typing import List

from RuleFinetune.RuleTrainer import Trainer
from utils.data import Rationale, Example, RuleBase, Rule
from utils.llm_models.call_openai import call_openai
from utils.read_datasets import read_datasets
from utils.sub_score import parse_response

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
    max_seq_len = 500
    for rule in sorted_rule_instance:
        sampled_rule_instance.append(rule)

        max_seq_len -= len(rule.content.split())
        if max_seq_len < 0:
            break

    sampled_rule_instance.sort(key=operator.attrgetter('content'))

    return sampled_rule_instance

def prompt_rules(rules: List[Rule]) -> str:
    """
    返回所有rules作为prompt

    下面有xx规则，优先从中找，找得到的话，格式是Existed Rule xxx；找不到就自己生成，生成的格式是Rulexxxx。
    write_rules(建议排序)

    加一个选择或者说删除机制，比如score=0的rule直接丢弃
    stages: 把所有的rule按分数排序分成3个stage
    proportion: 根据这个比例进行采样
    第n个阶段中采出rule_num*proportion的样放到rule_name里

    相关prompt部分已经被移除放到autoCoT里
    """
    # out = 'Instruction: '
    # out += 'For your guidance, here are several reference rules. '
    # out += 'Endeavor to adhere to these rules when responding to queries in <retrieved_rule>xxx<retrieved_rule> format. '
    # out += 'However, if adherence is unfeasible, you are permitted to establish your own rules in <new_rule>xxx<new_rule> format. '

    out = '''Instruction: Following are several existed knowledge in knowledge base. When you answer the questions, ''' \
          '''try to use the provided knowledge whenever possible by "we retrieve" prefix. ''' \
          '''Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to ''' \
          '''establish your own knowledge by "we add" prefix. '''

    out += '\nKnowledge base (involved rules):\n'
    out += '\n'.join([
        # str(idx+1)+':\t'+rn.content
        rn.content
        for idx, rn in enumerate(rules)])

    out += '\n\n'
    return out

def eval_step(example: Example, prompt_content: str):
    """
    :param prompt_content: 用var1;var2;...的格式prompt的组成
    """
    global rule_base
    added_rules = prompt_rules(sample_rule(rule_base))
    demo_prompt = '''Next, I will give you some examples to indicate how to use existed rules by  ''' \
                  '''"we retrieve" prefix and add the new rules into knowledge base by "we have" prefix.\nExamples: \n'''

    # prompt = added_rules + '\n' + 'Then, please answer this question: \n' + example.question + "\nAnswer: Let's think step by step."
    rules ='\n'.join([
        # str(idx+1)+':\t'+rn.content
        rn.content
        for idx, rn in enumerate(sample_rule(rule_base))])
    rule_prompt = """Here are several rules, followed by several examples. Then you are required to answer a new question.
Knowledge base:
{}\n\n""".format(rules)
    five_shot_textual = """Document: Anthony went to the park with his father, James. Annie took her uncle James to the grocery store. Alan and his daughter Annie spent Father's Day together. Annie took her dad out to a sports bar, and they had a great time watching football and drinking beer there.
Question: Anthony is Alan's what?
Answer: We first extract all triplets from the document. We then find the path from Alan to Anthony.
Finally, we reduce the relations on the path to get the answer.
The triplets include (Anthony, father, James), (Annie, uncle, James), (Alan, daughter, Annie).
The path from Alan to Anthony is (Alan, daughter, Annie), (Annie, uncle, James), (James, son, Anthony).
The relations on the path are daughter, uncle, son.
<rule>Siblings of a parent are aunts or uncles to their sibling's children<rule>. So the relations are reduced to brother, son.
<rule>The child of one's sibling is one's niece or nephew<rule>. So the relations are reduced to nephew.
Therefore, Anthony is Alan's nephew.

Document: Valerie's biggest accomplishment is raising her son Carlos. Annie does n't like having to babysit her younger brother, Emmanuel. Valerie and her son Emmanuel had lunch together at a local Chinese restaurant.
Question: Carlos is Annie's what?
Answer: We first extract all triplets from the document. We then find the path from Annie to Carlos.
Finally, we reduce the relations on the path to get the answer.
The triplets include (Valerie, son, Carlos), (Annie, brother, Emmanuel), (Valerie, son, Emmanuel).
The path from Annie to Carlos is (Annie, brother, Emmanuel), (Emmanuel, mother, Valerie), (Valerie,
son, Carlos). The relations on the path are brother, mother, son.
<rule>Siblings share the same parents<rule>. So the relations are reduced to mother, son.
<rule>Children of the same parents are siblings<rule>. So the relations are reduced to brother.
Therefore, Carlos is Annie's brother.

Document: James likes to take his daughter Jeanna fishing. James loves cooking with his daughter. Her name is Beverly. Jeanna loves visiting with her aunt Michelle.
Question: Michelle is Beverly's what?
Answer: We first extract all triplets from the document. We then find the path from Beverly to Michelle.
Finally, we reduce the relations on the path to get the answer.
The triplets include (James, daughter, Jeanna), (James, daughter, Beverly), (Jeanna, aunt, Michelle).
The path from Beverly to Michelle is (Beverly, father, James), (James, daughter, Jeanna), (Jeanna, aunt,
Michelle). The relations on the path are father, daughter, aunt.
<rule>Children of the same parents are siblings<rule>. So the relations are reduced to sister, aunt.
<rule>Siblings of a parent are aunts or uncles to their sibling's children<rule>. So the relations are reduced to aunt.
Therefore, Michelle is Beverly's aunt.

Document: Lee was finally coming of age and it was time for him and his father to go on a coming of age camping trip. Beverly, James's younger daughter, decided she wanted to go on the trip despite being several years younger. Jeanna took her younger sister Beverly to the carnival last weekend.
Question: Jeanna is Lee's what?
Answer: We first extract all triplets from the document. We then find the path from Lee to Jeanna.
Finally, we reduce the relations on the path to get the answer.
The triplets include (Lee, father, James), (James, daughter, Beverly), (Jeanna, sister, Beverly).
The path from Lee to Jeanna is (Lee, father, James), (James, daughter, Beverly), (Beverly, sister, Jeanna).
The relations on the path are father, daughter, sister.
<rule>Children of the same parents are siblings<rule>. So the relations are reduced to sister, sister.
<rule>Sibling relationship<rule>. So the relations are reduced to sister.
Therefore, Jeanna is Lee's sister.

Document: Craig's sister, Rosie, bought movie tickets at a discount rate. Rosie and her father Elliott love to go skiing. Often, Elliott will invite his mother Molly to join them.
Question: Molly is Craig's what?
Answer: We first extract all triplets from the document. We then find the path from Craig to Molly.
Finally, we reduce the relations on the path to get the answer.
The triplets include (Craig, sister, Rosie), (Rosie, father, Elliott), (Elliott, mother, Molly).
The path from Craig to Molly is (Craig, sister, Rosie), (Rosie, father, Elliott), (Elliott, mother, Molly).
The relations on the path are sister, father, mother.
<rule>Siblings share the same parents<rule>. So the relations are reduced to father, mother.
<rule>Grandparents' children are parents of their grandchildren<rule>. <rule>Mother's mother is the child's grandmother<rule>. So the relations are reduced to grandmother.
Therefore, Molly is Craig's grandmother.

Please answer the next new question:
{}
Answer:""".format(example.question).replace('Context: ', 'Document: ')

    five_shot_symbolic="""Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
Question: Anthony is Alan's what?
Answer:
The relation path is daughter, uncle, son (3)
The first relation pair is daughter and uncle. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children" from provided knowledge base, so daughter's uncle is brother, then the relations are reduced to brother, son (3 to 2).
The first relation pair is brother and son. We retrieve "The child of one's sibling is one's niece or nephew" from provided knowledge base, so brother's son is nephew, then the relations are reduced to nephew (2 to 1).
Therefore, the answer is nephew.

Context: The relations on the path from Annie to Carlos are brother, mother, son.
Question: Carlos is Annie's what?
Answer:
The relation path is brother, mother, son (3).
The first relation pair is brother's mother. We retrieve "Siblings share the same parents" from provided knowledge base, so brother's mother is mother, then the relations are reduced to mother, son (3 to 2).
The first relation pair is mother's son. We retrieve "Children of the same parents are siblings" from provided knowledge base, so mother's son is brother, then the relations are reduced to brother (2 to 1).
Therefore, the answer is brother.

Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
Question: Michelle is Beverly's what?
Answer:
The relation path is father, daughter, aunt (3).
The first relation pair is father's daughter. We retrieve "Children of the same parents are siblings" from provided knowledge base, so father's daughter is sister, then the relations are reduced to sister, aunt (3 to 2).
The first relation pair is sister's aunt. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children" from provided knowledge base, so sister's aunt is aunt, then the relations are reduced to aunt (2 to 1).
Therefore, the answer is aunt.

Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
Question: Jeanna is Lee's what?
Answer:
The relation path is father, daughter, sister (3).
The first relation pair is father's daughter. We retrieve "Children of the same parents are siblings" from provided knowledge base, so father's daughter is sister, then the relations are reduced to sister, sister (3 to 2).
The first relation pair is sister's sister. We have "sister's brother is brother" which is not in knowledge base, so sister's sister is sister, then the relations are reduced to sister (2 to 1).
Therefore, the answer is sister.

Context: The relations on the path from Craig to Molly are sister, father, mother.
Question: Molly is Craig's what?
Answer:
The relation path is sister, father, mother (3).
The first relation pair is sister's father. We retrieve "Siblings share the same parents" from provided knowledge base, so sister's father is father, then the relations are reduced to father, mother (3 to 2).
The first relation pair is father's mother. We retrieve "Grandparents' children are parents of their grandchildren" from provided knowledge base, so father's mother is grandmother, then the relations are reduced to grandmother (2 to 1).
Therefore, the answer is grandmother.

{}
Answer:""".format(example.question).replace('Context: ', 'Document: ')

#     """Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
# Question: Anthony is Alan's what?
# Answer:
# The relation path is daughter, uncle, son (3).
# The first relation pair is daughter's uncle, Because "Siblings of a parent are aunts or uncles to their sibling's children". So the relations are reduced to brother, son (3 to 2).
# The first relation pair is brother's son, Because "The child of one's sibling is one's niece or nephew". So the relations are reduced to nephew (2 to 1).
# Therefore, the answer is nephew.
#
# Context: The relations on the path from Annie to Carlos are brother, mother, son.
# Question: Carlos is Annie's what?
# Answer:
# The relation path is brother, mother, son (3).
# The first relation pair is brother's mother, Because "Siblings share the same parents". So the relations are reduced to mother, son (3 to 2).
# The first relation pair is mother's son, Because "Children of the same parents are siblings". So the relations are reduced to brother (2 to 1).
# Therefore, the answer is brother.
#
# Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
# Question: Michelle is Beverly's what?
# Answer:
# The relation path is father, daughter, aunt (3).
# The first relation pair is father's daughter, Because "Children of the same parents are siblings". So the relations are reduced to sister, aunt (3 to 2).
# The first relation pair is sister's aunt, Because "Siblings of a parent are aunts or uncles to their sibling's children". So the relations are reduced to aunt (2 to 1).
# Therefore, the answer is aunt.
#
# Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
# Question: Jeanna is Lee's what?
# Answer:
# The relation path is father, daughter, sister (3).
# The first relation pair is father's daughter, Because "Children of the same parents are siblings". So the relations are reduced to sister, sister (3 to 2).
# The first relation pair is sister's sister, Because "Sibling relationship". So the relations are reduced to sister (2 to 1).
# Therefore, the answer is sister.
#
# Context: The relations on the path from Craig to Molly are sister, father, mother.
# Question: Molly is Craig's what?
# Answer:
# The relation path is sister, father, mother (3).
# The first relation pair is sister's father, Because "Siblings share the same parents". So the relations are reduced to father, mother (3 to 2).
# The first relation pair is father's mother, Because "Grandparents' children are parents of their grandchildren". So the relations are reduced to grandmother (2 to 1).
# Therefore, the answer is grandmother.
#
# {}
# Answer:""".format(example.question.replace('Context', 'Document'))

#     + demos.replace('<retrieved_rule>', '<rule>').replace('<new_rule>', '<rule>') + '\n' + \
#     example.question + "\nAnswer:"
#     prompt = demos.replace('<retrieved_rule>', '<rule>').replace('<new_rule>', '<rule>') + '\n' + example.question + "\nAnswer:"
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
# Answer: Answer in the provided example, which must include the string ‘The answer is '""".format(example.question.strip())

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
#     prompt = eval(" + ".join(prompt_content.split(';')))
    prompt = five_shot_symbolic #added_rules + demo_prompt +
    print(prompt)
    response = call_openai(prompt)
    rationale = example.parse_response(response)
    prediction = Rationale.clean_prediction(rationale['prediction'])

    score = parse_response(question=example.question, response=response,
                           added_rules=rule_prompt)

    print(rationale, prediction, example.gold_label, score)


    return rationale, prediction, example.gold_label, score

prompt_content = 'five_shot_symbolic'
save_path = "./{0}6.txt".format(prompt_content)
with open(save_path, 'w', encoding="utf8") as f:
    pass

correct_cnt = 0
scores = []
final_dataset = test_dataset
with ThreadPoolExecutor(max_workers=200) as executor:
    futures = [executor.submit(eval_step, example, prompt_content) for example in final_dataset]
    for future in futures:
        rationale, prediction, gold_label, score = future.result()
        if prediction == gold_label:
            correct_cnt += 1

        with open(save_path, 'a', encoding="utf8") as f:
            f.write(str(rationale)+'\n')

        scores.append(score)

print(f"测试集上的准确率为：{correct_cnt / len(final_dataset)}")
print(scores)
print("=====================")
for key in ["step_relation_path", "length_reduce", "first_two_relations", "rule_retrieve", "step_inference_accuracy"]:
    success_key_score = [score[key] for score in scores if key in score]
    if len(success_key_score):
        print(key, sum(success_key_score) / len(success_key_score))