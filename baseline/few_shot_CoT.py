import argparse
import re
from concurrent.futures import ThreadPoolExecutor

from utils.data import Rationale, Example
from utils.llm_models.call_openai import call_openai
from utils.read_datasets import read_datasets

demos = """Context: Donald wanted to go visit his grandfather Jason. His dad Michael said okay and drove him to over to Jason house.
Question: Jason is Michael's what?
Answer: <rule> Children's grandparents are their parents' parents <rule> Given the context, Donald is visiting his grandfather Jason. Michael, Donald's father, is driving him there. According to the rule, if Jason is Donald's grandfather, then Jason must be Michael's father. Therefore, Jason is Michael's father. The answer is father.

Context: Michael is taking his son Donald out for coffee. Donald invited his uncle Christopher to dinner
Question: Christopher is Michael's what?
Answer: <rule> Brothers' children are your nieces or nephews. <rule> <rule> Your child's uncle is either your brother or your spouse's brother. <rule> Given the context: Michael is taking his son Donald out for coffee, and Donald invited his uncle Christopher to dinner. Applying the rules: Since Christopher is Donald's uncle, we apply the second rule: Your child's uncle is either your brother or your spouse's brother. Since the relationship is through Michael's son, we can infer that Christopher is Michael's brother. Therefore, Christopher is Michael's brother. The answer is brother.

Context: Christopher went out for pizza with his daughter Lucille and his brother Dwight.
Question: Lucille is Dwight's what?
Answer: <rule> Children of siblings are cousins to each other. <rule> <rule> Siblings are individuals who share at least one parent. <rule> <rule> The child of one's sibling is one's niece or nephew. <rule> Given the context: 1. Christopher is the father of Lucille. 2. Dwight is the brother of Christopher. Applying the rules: - Since Dwight is Christopher's brother, they are siblings (Rule 2). - Lucille, being the daughter of Christopher, is the child of Dwight's sibling (Christopher). - Therefore, Lucille is Dwight's niece (Rule 3). The answer is niece.

Context: Michael and his son Donald went to the store to by bread. Donald took his brother Russell out to get drinks after a long work week.
Question: Russell is Michael's what?
Answer: <rule> Children of the same parents are siblings to each other. <rule> <rule> If Person A is the parent of Person B, and Person B is a sibling of Person C, then Person A is the parent of Person C. <rule> From the context: 1. Donald is Michael's son. 2. Donald took his brother Russell out. Applying the rules: - Since Donald is Michael's son, and Russell is Donald's brother, we apply the first rule and acknowledge that Donald and Russell are siblings. - Applying the second rule, since Michael is the parent of Donald, and Donald is a sibling of Russell, it follows that Michael is also the parent of Russell. Answer: Russell is Michael's son. The answer is son.

Context: Jason went bike riding his grandson Joe. Joe's father Dwight was skateboarding.
Question: Jason is Dwight's what?
Answer: <rule> Parents of one's parents are called grandparents. <rule> <rule> Fathers are male parents. <rule> Since Joe is Jason's grandson, Jason must be the grandparent of Joe. Given that Dwight is Joe's father and therefore a male parent, we can deduce that Jason is Dwight's father or mother. Since the question asks specifically about Jason, who is involved in an activity with his grandson, it is most likely that Jason is male (although the information provided does not explicitly state Jason's gender). Assuming Jason is male: <rule> Male grandparents are called grandfathers. <rule> Therefore, Jason is Dwight's father. To summarize: Jason (grandfather of Joe) --> Dwight's father. The answer is father."""

demos = """Context: Dwight is one of Myrna's brothers. They have lots of fun together. Jose loved his mother, Myrna, very much.
Question: Dwight is Jose's what?
Answer:
Dwight is one of Myrna's brothers. <thus> Dwight is Myrna's brother.
Jose loved his mother, Myrna, very much. <thus> Jose is Myrna's son.
Dwight is Myrna's brother; Jose is Myrna's son. <thus> Dwight is Jose's uncle. <rule> If A is B's brother, C is B's son, then A is C's uncle. <rule>
Dwight is Jose's uncle.

Context: Alma met her husband Michael when they were just 12 years old. Jason and his son Michael went to the park to look for different kinds of birds.
Question: Jason is Alma's what?
Answer:
Alma met her husband Michael when they were just 12 years old. <thus> Alma is Michael's wife.
Jason and his son Michael went to the park to look for different kinds of birds. <thus> Jason is Michael's father.
Alma is Michael's wife; Jason is Michael's father. <thus> Alma is Jason's daughter-in-law. <rule> If A is B's wife, C is B's father, then A is C's daughter-in-law. <rule>
Alma is Jason's daughter-in-law. <thus> Jason is Alma's father-in-law. <rule> If A is B's daughter-in-law, then B is A's father-in-law. <rule>
Jason is Alma's father-in-law.

Context: Gabrielle bought her son Michael a basketball for his birthday. Asia called her dad Michael to pick her up from school.
Question: Gabrielle is Asia's what?
Answer:
Gabrielle bought her son Michael a basketball for his birthday. <thus> Michael is Gabrielle's son.
Asia called her dad Michael to pick her up from school. <thus> Michael is Asia's father.
Michael is Gabrielle's son; Michael is Asia's father. <thus> Gabrielle is Asia's grandmother. <rule> If A is B's son, A is C's father, B is female, then B is C's grandmother. <rule>
Gabrielle is Asia's grandmother.

Context: Michael and his son Donald went to the store to by bread. Donald took his brother Russell out to get drinks after a long work week.
Question: Russell is Michael's what?
Answer:
Michael and his son Donald went to the store to by bread. <thus> Donald is Michael's son.
Donald took his brother Russell out to get drinks after a long work week. <thus> Russell is Donald's brother.
Donald is Michael's son; Russell is Donald's brother. <thus> Russell is Michael's son. <rule> If A is B's son, C is B's brother, then C is A's son. <rule>
Russell is Michael's son.

Context: Myrna and her daughter Guillermina went out for ice cream. Myrna got a cone and Guillermina got a sundae. Michael, Guillermina's uncle, recently got into fishing.
Question: Michael is Myrna's what?
Answer:
Myrna and her daughter Guillermina went out for ice cream. <thus> Guillermina is Myrna's daughter.
Michael, Guillermina's uncle, recently got into fishing. <thus> Michael is Guillermina's uncle.
Guillermina is Myrna's daughter; Michael is Guillermina's uncle. <thus> Michael is Myrna's brother. <rule> If A is B's daughter, C is A's uncle, then C is B's brother. <rule>
Michael is Myrna's brother.

Now it's your turn, answer the question:"""

parser = argparse.ArgumentParser(description="Rule-Finetune")
args = parser.parse_args()
args.data_dir = "../data/CLUTRR"
args.dataset = "CLUTRR"

train_dataset, valid_dataset, test_dataset = read_datasets(args)

def eval_step(example: Example):
    # prompt = demos + '\n' + example.question + "\nAnswer:"# Let's think step by step."
#     """Strategy:
# 1. Identify the Starting and Ending Points: Determine the starting and ending individuals in the context (e.g., Lisa and Dorothy in Example 1).
# 2. List the Relationships in Sequence: Enumerate the relationships given in the context in the order they are mentioned (e.g., sister, sister).
# 3. Apply Relational Logic: For each consecutive pair of relationships, apply logical reasoning to deduce the overall relationship between the starting and ending points. This involves understanding how relationships compound (e.g., sister of a sister is still a sister).
# 4. Use Existing Knowledge Base Rules: Whenever possible, apply existing rules from the knowledge base to support the reasoning. These should be referenced in the format <retrieved_rule>xxx<retrieved_rule>.
# 5. Create New Rules if Necessary: If the existing rules are not sufficient to establish the relationship, create new rules. These should be clearly identified in the format <new_rule>xxx<new_rule>.
# 6. Formulate the Answer: Based on the reasoning and rules applied, formulate the answer to the question posed (e.g., determine that Dorothy is Lisa's sister).
# 7. Review for Consistency and Logic: Ensure that the answer is logically consistent with the context and the applied rules.
# 8. Document the Solution Process: Record the steps and rules used in the reasoning process for later analysis and improvement of the knowledge base.\n\n"""

    response = call_openai(prompt)

    rationale = example.parse_response(response)
    prediction = Rationale.clean_prediction(rationale['prediction'])
    print(rationale, prediction, example.gold_label)
    return prediction, example.gold_label

correct_cnt = 0

final_dataset = test_dataset
with ThreadPoolExecutor(max_workers=200) as executor:
    futures = [executor.submit(eval_step, example) for example in final_dataset]
    for future in futures:
        prediction, gold_label = future.result()
        if prediction == gold_label:
            correct_cnt += 1

print(f"测试集上的准确率为：{correct_cnt / len(final_dataset)}")