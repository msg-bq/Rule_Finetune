rule_instruction = '''Instruction: Following are several existed rules in knowledge base. When you answer the questions, ''' \
          '''try to use the provided rules whenever possible. ''' \
          '''Try not to invent knowledge by yourself unless necessary.'''\
            '''Knowledge base:\n'''

rule_instruction_HtT = """Instruction: When you answer the questions, try to use the provided knowledge whenever possible. Try
not to invent knowledge by yourself unless necessary.b
Knowledge:
Aunt’s sister is aunt.\nBrother’s aunt is aunt.\nBrother’s brother is brother.\nBrother’s daughter is niece.Brother’s father is father.\nBrother’s grandfather is grandfather.\nBrother’s grandmother is grandmother.Brother’s mother is mother.\nBrother’s sister is sister.\nBrother’s son is nephew.\nBrother’s uncle is uncle.Brother’s wife is sister-in-law.\nBrother-in-law’s daughter is niece.\nBrother-in-law’s father is father-inlaw.\nBrother-in-law’s mother is mother-in-law.\nBrother-in-law’s son is nephew.\nDaughter’s aunt is sister.Daughter’s brother is son.\nDaughter’s daughter is granddaughter.\nDaughter’s grandfather is father.\nDaughter’s grandmother is mother.\nDaughter’s husband is son-in-law.\nDaughter’s sister is daughter.\nDaughter’sson is grandson.\nDaughter’s uncle is brother.\nDaughter-in-law’s daughter is granddaughter.\nDaughter-inlaw’s son is grandson.\nFather’s brother is uncle.\nFather’s daughter is sister.\nFather’s father is grandfather.Father’s mother is grandmother.\nFather’s sister is aunt.\nFather’s son is brother.\nFather’s wife is mother.Granddaughter’s brother is grandson.\nGranddaughter’s father is son.\nGranddaughter’s mother is daughter.\nGranddaughter’s sister is granddaughter.\nGranddaughter’s uncle is son.\nGrandfather’s daughter isaunt.\nGrandfather’s son is uncle.\nGrandmother’s daughter is aunt.\nGrandmother’s son is uncle.\nGrandson’s brother is grandson.\nGrandson’s father is son.\nGrandson’s mother is daughter.\nGrandson’s sister isgranddaughter.\nGrandson’s uncle is son.\nHusband’s daughter is daughter.\nHusband’s father is father-inlaw.\nHusband’s granddaughter is granddaughter.\nHusband’s grandson is grandson.\nHusband’s mother ismother-in-law.\nHusband’s son is son.\nMother’s brother is uncle.\nMother’s daughter is sister.\nMother’sfather is grandfather.\nMother’s mother is grandmother.\nMother’s sister is aunt.\nMother’s son is brother.Nephew’s grandfather is father.\nNephew’s grandmother is mother.\nNephew’s sister is niece.\nNiece’sbrother is nephew.\nNiece’s uncle is brother.\nSelf’s brother is brother.\nSister’s brother is brother.\nSister’sdaughter is niece.\nSister’s father is father.\nSister’s grandfather is grandfather.\nSister’s grandmother isgrandmother.\nSister’s husband is brother-in-law.\nSister’s mother is mother.\nSister’s sister is sister.\nSister’sson is nephew.\nSister-in-law’s daughter is niece.\nSister-in-law’s father is father-in-law.\nSister-in-law’smother is mother-in-law.\nSister-in-law’s son is nephew.\nSon’s aunt is sister.\nSon’s brother is son.\nSon’sdaughter is granddaughter.\nSon’s grandfather is father.\nSon’s grandmother is mother.\nSon’s sister isdaughter.\nSon’s son is grandson.\nSon’s uncle is brother.\nSon’s wife is daughter-in-law.\nSon-in-law’s son isgrandson.\nStep-daughter’s grandmother is mother.\nUncle’s sister is aunt.\nWife’s brother is brother-in-law.Wife’s daughter is daughter.\nWife’s father is father-in-law.\nWife’s granddaughter is granddaughter.\nWife’sgrandson is grandson.\nWife’s mother is mother-in-law.\nWife’s son is son.
"""

CLUTRR_five_shot_symbolic = """Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
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

CLUTRR_five_shot_rule_symbolic="""Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
Question: Anthony is Alan's what?
Answer:
The relation path is daughter, uncle, son (3)
The first relation pair is daughter and uncle. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children", so daughter's uncle is brother, then the relations are reduced to brother, son (3 to 2).
The first relation pair is brother and son. We retrieve "The child of one's sibling is one's niece or nephew", so brother's son is nephew, then the relations are reduced to nephew (2 to 1).
Therefore, the answer is nephew.

Context: The relations on the path from Annie to Carlos are brother, mother, son.
Question: Carlos is Annie's what?
Answer:
The relation path is brother, mother, son (3).
The first relation pair is brother's mother. We retrieve "Siblings share the same parents", so brother's mother is mother, then the relations are reduced to mother, son (3 to 2).
The first relation pair is mother's son. We retrieve "Children of the same parents are siblings", so mother's son is brother, then the relations are reduced to brother (2 to 1).
Therefore, the answer is brother.

Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
Question: Michelle is Beverly's what?
Answer:
The relation path is father, daughter, aunt (3).
The first relation pair is father's daughter. We retrieve "Children of the same parents are siblings", so father's daughter is sister, then the relations are reduced to sister, aunt (3 to 2).
The first relation pair is sister's aunt. We retrieve "Siblings of a parent are aunts or uncles to their sibling's children", so sister's aunt is aunt, then the relations are reduced to aunt (2 to 1).
Therefore, the answer is aunt.

Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
Question: Jeanna is Lee's what?
Answer:
The relation path is father, daughter, sister (3).
The first relation pair is father's daughter. We retrieve "Children of the same parents are siblings", so father's daughter is sister, then the relations are reduced to sister, sister (3 to 2).
The first relation pair is sister's sister. We have "sister's sister is sister", so sister's sister is sister, then the relations are reduced to sister (2 to 1).
Therefore, the answer is sister.

Context: The relations on the path from Craig to Molly are sister, father, mother.
Question: Molly is Craig's what?
Answer:
The relation path is sister, father, mother (3).
The first relation pair is sister's father. We retrieve "Siblings share the same parents", so sister's father is father, then the relations are reduced to father, mother (3 to 2).
The first relation pair is father's mother. We retrieve "Grandparents' children are parents of their grandchildren", so father's mother is grandmother, then the relations are reduced to grandmother (2 to 1).
Therefore, the answer is grandmother.

{}
Answer:"""#.format(example.question).replace('Context: ', 'Document: ')

# 这个是HtT风格的five_shot
CLUTRR_five_shot_symbolic_HtT = """Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
Question: Anthony is Alan’s what?
Answer: For daughter’s uncle, we retrieve <daughter><uncle>daughter’s uncle is brother. So the relations are reduced to brother, son.
For brother’s son, we retrieve <brother><son>brother’s son is nephew. So the relations are reduced to nephew.
Therefore, Anthony is Alan's nephew.

Context: The relations on the path from Annie to Carlos are brother, mother, son.
Question: Carlos is Annie’s what?
Answer: For brother’s mother, we retrieve <brother><mother>brother’s mother is mother. So the relations are reduced to mother, son.
For mother’s son, we retrieve <mother><son>mother’s son is brother. So the relations are reduced to brother.
Therefore, Carlos is Annie's brother.

Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
Question: Michelle is Beverly’s what?
Answer: For father’s daughter, we retrieve <father><daughter>father’s daughter is sister. So the relations are reduced to sister, aunt.
For sister’s aunt, we retrieve <sister><aunt>sister’s aunt is aunt. So the relations are reduced to aunt.
Therefore, Michelle is Beverly's aunt.

Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
Question: Jeanna is Lee’s what?
Answer: For father’s daughter, we retrieve <father><daughter>father’s daughter is sister. So the relations are reduced to sister, sister.
For sister’s sister, we retrieve <sister><sister>Sister’s sister is sister. So the relations are reduced to sister.
Therefore, Jeanna is Lee's sister.

Context: The relations on the path from Craig to Molly are sister, father, mother.
Question: Molly is Craig’s what?
Answer: For sister’s father, we retrieve <sister><father>sister’s father is father. So the relations are reduced to father, mother.
For father’s mother, we retrieve <father><mother>father’s mother is grandmother. So the relations are reduced to grandmother.
Therefore, Molly is Craig's grandmother.

{}
Answer:"""#replace('Document: ', 'Context: ')
# """Context: The relations on the path from Alan to Anthony are daughter, uncle, son.
# Question: Anthony is Alan’s what?
# Answer: For daughter’s uncle, we retrieve <rule>Siblings of a parent are aunts or uncles to their sibling's children<rule>. So the relations are reduced to brother, son.
# For brother’s son, we retrieve <rule>The child of one's sibling is one's niece or nephew<rule>. So the relations are reduced to nephew.
# Therefore, Anthony is Alan's nephew.
#
# Context: The relations on the path from Annie to Carlos are brother, mother, son.
# Question: Carlos is Annie’s what?
# Answer: For brother’s mother, we retrieve <rule>Siblings share the same parents<rule>. So the relations are reduced to mother, son.
# For mother’s son, we retrieve <rule>Children of the same parents are siblings<rule>. So the relations are reduced to brother.
# Therefore, Carlos is Annie's brother.
#
# Context: The relations on the path from Beverly to Michelle are father, daughter, aunt.
# Question: Michelle is Beverly’s what?
# Answer: For father’s daughter, we retrieve <rule>Children of the same parents are siblings<rule>. So the relations are reduced to sister, aunt.
# For sister’s aunt, we retrieve <rule>Siblings of a parent are aunts or uncles to their sibling's children<rule>. So the relations are reduced to aunt.
# Therefore, Michelle is Beverly's aunt.
#
# Context: The relations on the path from Lee to Jeanna are father, daughter, sister.
# Question: Jeanna is Lee’s what?
# Answer: For father’s daughter, we retrieve <rule>Children of the same parents are siblings<rule>. So the relations are reduced to sister, sister.
# For sister’s sister, we retrieve <rule>Sibling relationship<rule>. So the relations are reduced to sister.
# Therefore, Jeanna is Lee's sister.
#
# Context: The relations on the path from Craig to Molly are sister, father, mother.
# Question: Molly is Craig’s what?
# Answer: For sister’s father, we retrieve <rule>Siblings share the same parents<rule>. So the relations are reduced to father, mother.
# For father’s mother, we retrieve <rule>Grandparents' children are parents of their grandchildren<rule>. <rule>Mother's mother is the child's grandmother<rule>. So the relations are reduced to grandmother.
# Therefore, Molly is Craig's grandmother.
#
# {}
# Answer:"""#replace('Document: ', 'Context: ')

five_shot_HtT = """"""

LANG_rule_instruction = '''Instruction: Following are several existed rules in knowledge base. When you correct the grammatical eroors, ''' \
                        '''try to use them whenever possible. Not to invent knowledge by yourself unless necessary. ''' \
                        '''Try to keep the words and punctuation of the original sentence, and correct only necessary grammatical errors \n'''\
                        '''Knowledge base:\n'''

LANG_CoT = """Sentence: December
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors. 
So the revised sentence is "December".

Sentence: We excited the show !
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error 1: excited -> were excited at
So the revised sentence is "We were excited at the show !"

Sentence: I asked her if she have boyfriend or not .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error 1: have -> had
Error 2: boyfriend -> a boyfriend
Error 3: or not -> ""
So the revised sentence is "I asked her if she had a boyfriend ."

Sentence: Crossing my fingers for them ^ _ ^ ! ! ! lol
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
So the revised sentence is "Crossing my fingers for them ^ _ ^ ! ! ! lol"

Sentence: T `` .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
So the revised sentence is "T `` ."
"""

LANG_CoT_rule = """Sentence: December
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
So the revised sentence is "December".

Sentence: We excited the show !
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error 1: we retrieve "Subject-Verb agreement rule", so "excited -> were excited at"
So the revised sentence is "We were excited at the show !"

Sentence: I asked her if she have boyfriend or not .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error 1: we retrieve "Rule: Use the correct verb tense to match the timing of the action being described.", so "have -> had"
Error 2: we retrieve "Use "a" before specific instances of a noun, such as "a diary.", so "boyfriend -> a boyfriend"
Error 3: we retrieve "Including "or not" after "if" may occur in casual speech but is typically avoided in formal writing and speech.", so "or not -> ""
So the revised sentence is "I asked her if she had a boyfriend ."

Sentence: Crossing my fingers for them ^ _ ^ ! ! ! lol
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
So the revised sentence is "Crossing my fingers for them ^ _ ^ ! ! ! lol"
"""

dataset_prompt = {'CLUTRR': {'rule_instruction': rule_instruction,
                             'rule_instruction_HtT': rule_instruction_HtT,
                     # 'zero-shot': CLUTRR_zero_shot_symbolic,
                             'CoT': CLUTRR_five_shot_symbolic,
                             'CoT_rule': CLUTRR_five_shot_rule_symbolic,
                             'CoT_HtT': CLUTRR_five_shot_symbolic_HtT,},
                  'LANG_8': {'rule_instruction': LANG_rule_instruction,
                             'CoT': LANG_CoT,
                             'CoT_rule': LANG_CoT_rule}}