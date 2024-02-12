
default1 = """Answer: Let's think step by step. First rationale then answer."""\
           """If you use any prior knowledge or rule during the inference, write them briefly in "<Begin>xxx</End>" format. Only do this if you find them. """ \
           """Note that these rules should be true in general."""

default2 = """Answer: Let's think step by step. First rationale then answer."""\
           """If you use any rules during the inference, write rules in "<Begin>xxx</End>" format. Only do this if you find rules. """ \
           """Note that these rules should be true in general and concise."""

default3 =  """If you use some prior knowledge in the reasoning process, please surround the complete but concise meaning of knowledge with tag <Begin>xxx</End> individually. """\
"""Note that these knowledge should be universally applicable, including objective truth, laws of nature, universal rules and so on."""

HtT_version = """Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
Question: Anthony is Alan’s what?
Answer:
For daughter’s uncle, we have daughter’s uncle is brother. So the relations are reduced to brother, son.
For brother’s son, we have brother’s son is nephew. So the relations are reduced to nephew.
Therefore, the answer is nephew.

Context: The relations on the path from Annie to Carlos are brother, mother, son.
Question: Carlos is Annie’s what?
Answer:
For brother’s mother, we have brother’s mother is mother. So the relations are reduced to mother, son.
For mother’s son, we have mother’s son is brother. So the relations are reduced to brother.
Therefore, the answer is brother.

Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
Question: Michelle is Beverly’s what?
Answer:
For father’s daughter, we have father’s daughter is sister. So the relations are reduced to sister, aunt.
For sister’s aunt, we have sister’s aunt is aunt. So the relations are reduced to aunt.
Therefore, the answer is aunt.

Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
Question: Jeanna is Lee’s what?
Answer:
For father’s daughter, we have father’s daughter is sister. So the relations are reduced to sister, sister.
For sister’s sister, we have sister’s sister is sister. So the relations are reduced to sister.
Therefore, the answer is sister.

Context: The relations on the path from Craig to Molly are sister, father, mother.
Question: Molly is Craig’s what?
Answer:
For sister’s father, we have sister’s father is father. So the relations are reduced to father, mother.
For father’s mother, we have father’s mother is grandmother. So the relations are reduced to grandmother.
Therefore, the answer is grandmother."""

cot_trigger = {'default1': default1, 'default2': default2, 'default3': default3, 'HtT_version': HtT_version}

# -----------------------------------------------------------------

pred_trigger = {"CLUTRR": "The answer is",
                "LANG_8": "The revised grammatically correct sentence is",
                "STS_B": "The sentiment of the review is"}

#--------------------------------------------------------------------
HtT_train_prompt = """When you answer the questions, try to use the provided knowledge whenever possible. Try not to invent knowledge by yourself unless necessary."""
default1_train_prompt = """Instruction: Following are several existed knowledge in knowledge base. When you answer the questions, """ \
          """try to use the provided knowledge whenever possible in <retrieved_rule>xxx<retrieved_rule> format. """ \
          """Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to """ \
          """establish your own rules in <new_rule>xxx<new_rule> format.\n"""

train_prompt = {'default1': default1_train_prompt, 'HtT_version': HtT_train_prompt}