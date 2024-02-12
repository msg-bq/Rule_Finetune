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

LANG_rule_instruction = '''When correcting grammatical errors in existing rules from the knowledge base, utilize the rules whenever possible. Only invent knowledge when necessary.\n'''\
                        '''Note that you should only correct essential grammatical errors, and maintain the original punctuations as much as possible, it's no need to delete or add the optional words and punctuations.'''\
                        '''\nKnowledge base:\n'''
#I will use "✓" and "✗" to indicate the confidence degree of the knowledge base rules. "✓" means high confidence and "✗" means low confidence.
#In the other hand, remember and avoid to use the knowledge with low confidence.
#proven knowledge with confidence
LANG_rule_instruction = """Following knowledge base involves some proven knowledge that you can use to correct the grammatical errors. Utilize the knowledge base whenever possible. Only introduce your intrinsic knowledge when necessary.
Note: You should use those knowledge with high confidence (which means they are more likely to be correct) to overriding implicit knowledge in your model. 

Koweledge base:
"""

# LANG_rule_instruction = """You are now operating as an AI model named "GrammarMaster" or GM. Your task is to utilize the provided knowledge base to correct the grammatical errors in the given text. Use the rules with high confidence to override semantic priors in your mind. Only use your intrinsic knowledge when necessary.
# Your primary aim is to maintain the original words and punctuation as much as possible, and only correct essential grammatical errors in the text.
# We're counting on you to apply the proven rules effectively to enhance the overall grammar of the given content."""

# LANG_rule_instruction = """Use the provided knowledge base to correct any essential grammatical errors in your responses. Utilize the rules with high confidence from the knowledge base to override semantic priors in your model. Only introduce your intrinsic knowledge when necessary. Maintain the original words and punctuation as much as possible and only correct essential grammatical errors. When answering questions, ensure to incorporate the knowledge base rules whenever possible for accurate and grammatically correct responses."""

LANG_CoT = """Maintain the original words and punctuation as much as possible, and only correct minimal and essential grammatical errors.
Again, you should only correct minimal and essential grammatical errors, but maintain the original punctuations as much as possible, it's no need to delete or add the optional words and punctuations.

Sentence: December
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors. 
Therefore, the revised sentence is "December".


Sentence: We excited the show !
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error1: "excited" → "were excited"
Error2: "excited" → "excited at"
Therefore, the revised sentence is "We were excited at the show !"


Sentence: I asked her if she have boyfriend or not .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: Error1: "have" → "had"
Error2: "boyfriend" → "a boyfriend"
Error3: "or not" → ""
Therefore, the revised sentence is "I asked her if she had a boyfriend ."


Sentence: Crossing my fingers for them ^ _ ^ ! ! ! lol
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
Therefore, the revised sentence is "Crossing my fingers for them ^ _ ^ ! ! ! lol"


Sentence: T `` .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: No grammar errors.
Therefore, the revised sentence is "T `` ."
"""


LANG_CoT_rule = """Sentence: December
Question: What's the grammar errors and revised sentence of above sentence?
Answer: First identify the sentence structure: Determine whether the sentence is declarative, interrogative, imperative, or exclamatory to understand its purpose and structure.
Then, check for subject-verb agreement, verb tense, pronoun usage, set phrase, article and determiner accuracy, prepositions, spelling and capitalization, but try to maintain the original punction.

Analyse process:
No grammar errors.
Therefore, the revised sentence is "December".


Sentence: We excited the show !
Question: What's the grammar errors and revised sentence of above sentence?
Answer: First identify the sentence structure: Determine whether the sentence is declarative, interrogative, imperative, or exclamatory to understand its purpose and structure.
Then, check for subject-verb agreement, verb tense, pronoun usage, set phrase, article and determiner accuracy, prepositions, spelling and capitalization, but try to maintain the original punction.

Analyse process:
Error1: subject-verb agreement
Rule1: "Subject-Verb agreement rule". 
Adjustment1: Change "excited" to "were excited"

Error2: prepositions
Rule2: "Use the correct preposition to indicate the relationship between words in a sentence."
Adjustment2: change "excited" to "excited at"

Therefore, the revised sentence is "We were excited at the show !"


Sentence: I asked her if she have boyfriend or not .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: First identify the sentence structure: Determine whether the sentence is declarative, interrogative, imperative, or exclamatory to understand its purpose and structure.
Then, check for subject-verb agreement, verb tense, pronoun usage, set phrase, article and determiner accuracy, prepositions, spelling and capitalization, but try to maintain the original punction.

Analyse process with knowledge base:
Error1: verb tense
Rule1: "Use the correct verb tense to match the timing of the action being described."
Adjustment1: Change "have" to "had"

Error2: article and determiner accuracy
Rule2: "Use "a" before specific instances of a noun, such as "a diary."
Adjustment2: Change "boyfriend" to "a boyfriend"

Error3: set phrase
Rule3: "Including "or not" after "if" may occur in casual speech but is typically avoided in formal writing and speech."
Adjustment3: Remove "or not"

Therefore, the revised sentence is "I asked her if she had a boyfriend ."


Sentence: Crossing my fingers for them ^ _ ^ ! ! ! lol
Question: What's the grammar errors and revised sentence of above sentence?
Answer: First identify the sentence structure: Determine whether the sentence is declarative, interrogative, imperative, or exclamatory to understand its purpose and structure.
Then, check for subject-verb agreement, verb tense, pronoun usage, set phrase, article and determiner accuracy, prepositions, spelling and capitalization, but try to maintain the original punction.

Analyse process:
No grammar errors.
Therefore, the revised sentence is "Crossing my fingers for them ^ _ ^ ! ! ! lol"


Sentence: T `` .
Question: What's the grammar errors and revised sentence of above sentence?
Answer: First identify the sentence structure: Determine whether the sentence is declarative, interrogative, imperative, or exclamatory to understand its purpose and structure.
Then, check for subject-verb agreement, verb tense, pronoun usage, set phrase, article and determiner accuracy, prepositions, spelling and capitalization, but try to maintain the original punction.

Analyse process:
No grammar errors.
Therefore, the revised sentence is "T `` ."
"""

LANG_CoT_rule = LANG_CoT

# ===================================
# SST2_CoT = """Overall procedure: First, understand the sentence and identify key concepts. Second, consider the context of the sentence. Determine the sentiment based on the meaning of key words and sentence context.
#
# Review: will find little of interest in this film , which is often preachy and poorly acted
# Question: What's the sentiment of above review?
# Answer: First, the sentence is a critique of a film. Key phrases include: "little ofl interest", "preachy", and "poorly acted". Second, the sentence is reviewing a film. All key phrases carry negative connotations. The sentiment is negative.
#
# Review: gorgeous and deceptively minimalist
# Question: What's the sentiment of above review?
# Answer: First, the key phrases in the sentence are "gorgeous" and "deceptively minimalist". Second, the sentence is describing something's appearance or style. "Gorgeous" is positive, and "deceptively minimalist" suggests a surprising or unexpected simplicity, which can be seen as positive. The sentiment is positive.
#
# Review: that would make lesser men run for cover .
# Question: What's the sentiment of above review?
# Answer: First, the key phrases in the sentence are "lesser men" and "run for cover". Second, the context implies a situation or task that is challenging or intimidating. The overarching sentiment celebrates the bravery, resilience, or capability of the personbeing described. The sentiment is positive.
#
# Review: of marivaux 's rhythms , and mira sorvino 's limitations as a classical actress
# Question: What's the sentiment of above review?
# Answer: First, the key phrases in the sentence are "marivaux's rhythms" and "mira sorvino's limitations as a classical actress". Second, the context seems to comment on the style of Marivaux and the abilities of Mira Sorvino. The term "limitations" indicates a shortfall or inadequacy in Mira Sorvino's acting in classical roles. The sentiment is negative.
#
# Review: viewers of barney 's crushingly self-indulgent spectacle
# Question: What's the sentiment of above review?
# Answer: First, the key phrases in the sentence are "barney's" and "crushingly self-indulgent spectacle". Second, the context appears to be a commentary on a spectacle associated with Barney. The term "crushingly self-indulgent" implies excessive self-interest and a lack of consideration for the audience. The sentiment is negative.
# """

SST2_CoT = """Overall procedure: First, understand the sentence and identify key concepts. Second, consider the context of the sentence. Determine the sentiment based on the meaning of key words and sentence context.

Q: For the sentence "will find little of interest in this film", is the sentiment in this sentence positive or negative?
A: First, the sentence is a critique of a film. Key phrases include: "little of interest", "preachy", and "poorly acted". Second, the sentence is reviewing a film. All key phrases carry negative connotations. The sentiment is negative.

Q: For the sentence "gorgeous and deceptively minimalist", is the sentiment in this sentence positive or negative?
A: First, the key phrases in the sentence are "gorgeous" and "deceptively minimalist". Second, the sentence is describing something's appearance or style. "Gorgeous" is positive, and "deceptively minimalist" suggests a surprising or unexpected simplicity, which can be seen as positive. The sentiment is positive.

Q: For the sentence "that would make lesser men run for cover", is the sentiment in this sentence positive or negative?
A: First, the key phrases in the sentence are "lesser men" and "run for cover". Second, the context implies a situation or task that is challenging or intimidating. The overarching sentiment celebrates the bravery, resilience, or capability of the personbeing described. The sentiment is positive.

Q: For the sentence "of marivaux's rhythms, and mira sorvino's limitations as a classical actress", is the sentiment in this sentence positive or negative?
A: First, the key phrases in the sentence are "marivaux's rhythms" and "mira sorvino's limitations as a classical actress". Second, the context seems to comment on the style of Marivaux and the abilities of Mira Sorvino. The term "limitations" indicates a shortfall or inadequacy in Mira Sorvino's acting in classical roles. The sentiment is negative.

Q: For the sentence "viewers of barney 's crushingly self-indulgent spectacle", is the sentiment in this sentence positive or negative?
A: First, the key phrases in the sentence are "barney's" and "crushingly self-indulgent spectacle". Second, the context appears to be a commentary on a spectacle associated with Barney. The term "crushingly self-indulgent" implies excessive self-interest and a lack of consideration for the audience. The sentiment is negative.
"""

SST2_rule_instruction = """Following knowledge base involves some proven knowledge that you can use to analyse the sentiment of given sentence. Utilize the knowledge base whenever possible. Only introduce your intrinsic knowledge when necessary.
Note: You should use those knowledge with high confidence (which means they are more likely to be correct) to overriding implicit knowledge in your model.

Koweledge base:
"""

SST2_CoT_rule = SST2_CoT


dataset_prompt = {'CLUTRR': {'rule_instruction': rule_instruction,
                             'rule_instruction_HtT': rule_instruction_HtT,
                     # 'zero-shot': CLUTRR_zero_shot_symbolic,
                             'CoT': CLUTRR_five_shot_symbolic,
                             'CoT_rule': CLUTRR_five_shot_rule_symbolic,
                             'CoT_HtT': CLUTRR_five_shot_symbolic_HtT,},
                  'LANG_8': {'rule_instruction': LANG_rule_instruction,
                             'CoT': LANG_CoT,
                             'CoT_rule': LANG_CoT_rule},
                  'STS_B': {'CoT': SST2_CoT,
                            'rule_instruction': SST2_rule_instruction,
                            'CoT_rule': SST2_CoT_rule}}