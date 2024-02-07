import re

debugging = 0

height = 0
width = 1
gender = 2
sibling = 3
selfSpouse = 4
parentSpouse = 5
siblingSpouse = 6
childSpouse = 7
otherSpouse = 8
valid = 9

answers = [[None]*100] * 100
genders = [None] * 100

input1 = ""
myanswer = ""
myanswer2 = ""

# The genders array prevents us from making the following error:
#    father's father's daughter = aunt and could also be mother
#    the correct answer is simply aunt -- mother is incorrect

def Spouse(sex):
    global input1
    global debugging, height, width, genders, sibling, selfSpouse, parentSpouse, siblingSpouse, childSpouse, otherSpouse, valid
    input = input1
    if (input != ""):
        input += "'s "

        if (sex == 'm'):
            input += "husband"
        elif (sex == 'f'):
            input += "wife"
        else:  # sex is ''
            input += "spouse"

        input1 = input

        origLength = len(answers)
        for i in range(origLength):
            if (answers[i][childSpouse] or answers[i][siblingSpouse] or answers[i][otherSpouse] or
                    (answers[i][selfSpouse] and answers[i][height] == 0) and answers[i][width] == 0):
                # can't go any further
                answers[i][valid] = 0  # indicates it is no longer valid
                continue

            answers[i][gender] = sex
            if (answers[i][height] == 0 and answers[i][width] == 0):  # self
                answers[i][selfSpouse] = 1
            elif (answers[i][height] > 0 and answers[i][width] == 0):  # parent
                answers[i][parentSpouse] = 1
            elif (answers[i][height] == 0 and answers[i][width] == 1):  # sibling
                answers[i][siblingSpouse] = 1
            elif (answers[i][height] < 0 and answers[i][width] == 0):  # child
                answers[i][childSpouse] = 1
            else:
                answers[i][otherSpouse] = 1

            w = answers[i][width]
            if (w == 0):
                h = answers[i][height]
                genders[h] = sex

        # @@@@@

        Display()


def Ordinal(i):
    if (i < 10 or i > 20):
        if (i % 10 == 1):
            return "1st"
        elif (i % 10 == 2):
            return "2nd"
        elif (i % 10 == 3):
            return "3rd"

    return i + "th"


def Times(i):
    if (i == 1):
        return "once"
    elif (i == 2):
        return "twice"
    else:
        return i + " times"


def Great(i):
    result = ""
    for j in range(i):
        result += "great "

    return result


def Grand(i):
    result = ""
    for j in range(i):
        if (result != ""):
            result += " "

        if (j != i - 1):
            result += "great"
        else:
            result += "grand"

    return result


def DebugSuffix(i):
    global answers, input1, myanswer, myanswer2
    global debugging, height, width, genders, sibling, selfSpouse, parentSpouse, siblingSpouse, childSpouse, otherSpouse, valid

    g = answers[i][gender]
    if (g == ""):
        g = "?"

    answer = "(" + \
    answers[i][height] + "," + answers[i][width] + "," + g + "," + answers[i][sibling] + "," + \
    answers[i][selfSpouse] + "," + answers[i][parentSpouse] + "," + \
    answers[i][siblingSpouse] + "," + answers[i][childSpouse] + "," + \
    answers[i][otherSpouse] + "," + answers[i][valid] + ") ("


    for j in range(-5, 6):
        if (genders[j] != None):
            answer += str(j) + ":" + str(genders[j]) + "|"

    answer += ")"
    return answer


def Display():
    # convert array of triplets to displayable text
    global answers, input1, myanswer, myanswer2
    global debugging, height, width, genders, sibling, selfSpouse, parentSpouse, siblingSpouse, childSpouse, otherSpouse, valid

    answer = ""
    for i in range(len(answers)):

        # extract the values in this triplet

        h = answers[i][height]
        w = answers[i][width]
        g = answers[i][gender]
        sbl = answers[i][sibling]
        ss = answers[i][selfSpouse]
        ps = answers[i][parentSpouse]
        sbs = answers[i][siblingSpouse]
        cs = answers[i][childSpouse]
        os = answers[i][otherSpouse]
        v = answers[i][valid]

        # ignore duplicate entries

        duplicate = 0
        for j in range(i):
            if (answers[j][height] == h and answers[j][width] == w and answers[j][gender] == g and answers[j][
                sibling] == sbl and
                    answers[j][selfSpouse] == ss and answers[j][parentSpouse] == ps and
                    answers[j][siblingSpouse] == sbs and answers[j][childSpouse] == cs and
                    answers[j][otherSpouse] == os and answers[j][valid] == v):
                duplicate = 1
                break

        if (duplicate):
            continue

        # separate alternates by commas

        if (answer != ""):
            answer += ", "

        if (not v):
            answer += "???"
            if (debugging):
                answer += DebugSuffix(i)

            continue

        if (h == 0):  # same generation
            if (w == 0):
                if (input1 != ""):
                    if (ss):
                        if (g == 'm'):
                            answer += "husband"
                        elif (g == 'f'):
                            answer += "wife"
                        else:
                            answer += "spouse"

                    else:
                        answer += "self"


            elif (w == 1):
                if (ps):
                    answer += "step "

                if (g == 'm'):
                    answer += "brother"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", brother"

                elif (g == 'f'):
                    answer += "sister"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", sister"

                else:
                    answer += "sibling"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", sibling"

                if (ss or sbs):
                    answer += " in law"

            else:
                answer += Ordinal(w - 1) + " cousin"

        elif (h > 0):  # relative is on higher generation
            if (ps):
                answer += "step "

            if (w == 0):
                if (g == 'm'):
                    answer += Grand(h - 1) + "father"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "father"

                elif (g == 'f'):
                    answer += Grand(h - 1) + "mother"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "mother"

                else:
                    answer += Grand(h - 1) + "parent"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "parent"

                if (ss):
                    answer += " in law"

            elif (w == 1):
                if (g == 'm'):
                    answer += Great(h - 1) + "uncle"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Great(h - 1) + "uncle"

                elif (g == 'f'):
                    answer += Great(h - 1) + "aunt"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Great(h - 1) + "aunt"

                else:
                    answer += Great(h - 1) + "uncle/aunt"
                    if (ps):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Great(h - 1) + "uncle/aunt"


            else:
                answer += Ordinal(w - 1) + " cousin " + Times(h) + " removed"
                if (ps):
                    if (debugging):
                        answer += DebugSuffix(i)

                    answer += ", " + Ordinal(w - 1) + " cousin " + Times(h) + " removed"



        else:  # h < 0, relative is on lower generation
            h = -h
            if (w == 0):
                if (ss):
                    answer += "step "

                if (g == 'm'):
                    answer += Grand(h - 1) + "son"
                    if (ss):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "son"

                elif (g == 'f'):
                    answer += Grand(h - 1) + "daughter"
                    if (ss):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "daughter"

                else:
                    answer += Grand(h - 1) + "child"
                    if (ss):
                        if (debugging):
                            answer += DebugSuffix(i)

                        answer += ", " + Grand(h - 1) + "child"

                if (cs):
                    answer += " in law"

            elif (w == 1):
                if (g == 'm'):
                    answer += Great(h - 1) + "nephew"
                elif (g == 'f'):
                    answer += Great(h - 1) + "niece"
                else:
                    answer += Great(h - 1) + "nephew/niece"

            else:
                answer += Ordinal(w - 1) + " cousin " + Times(h) + " removed"

        if (debugging):
            answer += DebugSuffix(i)

    comma = answer.index(", ") if ", " in answer else -1
    if (comma == -1):
        myanswer = answer
        myanswer2 = ""
    else:
        myanswer = answer[0:comma]
        myanswer2 = answer[comma + 2:]

    if (myanswer != ""):
        if (answers[0][selfSpouse] or answers[0][parentSpouse] or answers[0][siblingSpouse] or
                answers[0][childSpouse] or answers[0][otherSpouse]):
            myanswer += "  (non-blood relative)"
        else:
            myanswer += "  (blood relative)"


def NewAnswer(i):
    global debugging, height, width, genders, sibling, selfSpouse, parentSpouse, siblingSpouse, childSpouse, otherSpouse, valid

    # answers[i][height] = 0
    # answers[i][width] = 0
    # answers[i][height] = 0
    # answers[i][gender] = ''
    # answers[i][sibling] = '0'
    # answers[i][selfSpouse] = 0
    # answers[i][parentSpouse] = 0
    # answers[i][siblingSpouse] = 0
    # answers[i][childSpouse] = 0
    # answers[i][otherSpouse] = 0
    # answers[i][valid] = 1

    if len(answers) <= i:
        answers.extend([[]]*(i-len(answers)+1))
    answers[i].append(0)
    answers[i].append(0)
    answers[i].append('')
    answers[i].append('0')
    answers[i].append(0)
    answers[i].append(0)
    answers[i].append(0)
    answers[i].append(0)
    answers[i].append(0)
    answers[i].append(1)

def Clear():
    global input1, answers, genders
    input1 = ""
    answers = []
    genders = [""] * 100
    NewAnswer(0)
    genders[0] = ''
    Display()


def Backspace():
    inputArray = input1.split("'s ")
    Clear()
    for i in range(len(inputArray)-1):  # note we are skipping last array element
        relation = inputArray[i]
        if (relation == "father"):
            Calculate(1, 'm')
        elif (relation == "parent"):
            Calculate(1, '')
        elif (relation == "mother"):
            Calculate(1, 'f')
        elif (relation == "brother"):
            Calculate(0, 'm')
        elif (relation == "sibling"):
            Calculate(0, '')
        elif (relation == "sister"):
            Calculate(0, 'f')
        elif (relation == "son"):
            Calculate(-1, 'm')
        elif (relation == "child"):
            Calculate(-1, '')
        elif (relation == "daughter"):
            Calculate(-1, 'f')
        elif (relation == "husband"):
            Spouse('m')
        elif (relation == "wife"):
            Spouse('f')
        elif (relation == "spouse"):
            Spouse('')
        else:
            raise Exception("invalid relation: " + relation)


def Calculate(delta, sex):
    # display keystrokes that the user typed
    global input1
    input = input1
    if (input != ""):
        input += "'s "

    if (delta == 1):
        if (sex == 'm'):
            input += "father"
        elif (sex == 'f'):
            input += "mother"
        else:  # sex is ''
            input += "parent"

    elif (delta == 0):
        if (sex == 'm'):
            input += "brother"
        elif (sex == 'f'):
            input += "sister"
        else:  # sex is ''
            input += "sibling"

    else:  # delta is -1
        if (sex == 'm'):
            input += "son"
        elif (sex == 'f'):
            input += "daughter"
        else:  # sex is ''
            input += "child"

    input1 = input

    # calculate the relationship as a n-tuple
    global answers
    origLength = len(answers)
    for i in range(origLength):
        answers[i][gender] = sex

        h = answers[i][height]
        w = answers[i][width]
        g = answers[i][gender]
        sbl = answers[i][sibling]
        ss = answers[i][selfSpouse]
        ps = answers[i][parentSpouse]
        sbs = answers[i][siblingSpouse]
        cs = answers[i][childSpouse]
        os = answers[i][otherSpouse]

        if (cs or sbs or os):  # can't get any further relationships
            answers[i][valid] = 0  # indicates it is no longer valid
            continue

        if (delta == 1):
            h += 1
            answers[i][height] = h
            if (h == 0 and w == 0):  # e.g., child's parent
                # came from child back to self, so assume it is my spouse
                answers[i][selfSpouse] = 1
                if (genders[h] == "" or g == "" or genders[h] == g):
                    length = len(answers)
                    answers.append([])
                    NewAnswer(length)  # and it could be myself
                    answers[length][gender] = genders[h] if g == "" else g

            elif (h == 0 and w == 1):  # e.g., sibling's child's parent
                if (genders[h] != "" and g != "" and genders[h] != g):
                    answers[i][siblingSpouse] = 1

            elif (h < 0):  # e.g., child's child's parent
                if (genders[h] == "" or g == "" or genders[h] == g):
                    answers[i][gender] = genders[h] if (g == "") else g
                    length = len(answers)
                    answers.append([])
                    NewAnswer(length)
                    answers[length][height] = h
                    answers[length][childSpouse] = 1
                    if (g != ""):
                        answers[length][gender] = g
                    elif (genders[h] == 'm'):
                        answers[length][gender] = 'f'
                    elif (genders[h] == 'f'):
                        answers[length][gender] = 'm'
                    else:
                        answers[length][gender] = ""

                else:
                    answers[i][childSpouse] = 1

            if (w > 0 and h > 0):
                w -= 1
                answers[i][width] = w

        elif (delta == -1):
            h -= 1
            answers[i][height] = h
            if (h >= 0):  # e.g., parent's parent's child
                w += 1
                answers[i][width] = w
                if (w == 1 and (genders[h] == "" or g == "" or genders[h] == g)):
                    length = len(answers)
                    answers.append([])
                    NewAnswer(length)
                    answers[length][height] = h
                    answers[length][parentSpouse] = ps
                    answers[length][gender] = genders[h] if g == "" else g


        else:  # delta is 0
            if (h >= 0):  # e.g., parent's or self's sibling
                if (w == 0):
                    if (h == 0):  # self's sibling
                        answers[i][sibling] = 1

                    w = 1
                    answers[i][width] = w
                elif (w == 1 and (genders[h] == "" or g == "" or genders[h] == g)):
                    length = len(answers)
                    answers.append([])
                    NewAnswer(length)
                    answers[length][height] = h
                    answers[length][parentSpouse] = ps
                    answers[length][gender] = genders[h] if (g == "") else g

        if (w == 0 or h <= 0):
            #          if (w == 0 andand genders[h] == undefined) :
            #          if (w == 0 andand genders[h] != "") :
            genders[h] = g

def kinship_calculate(input_text):
    global myanswer

    inputArray = input_text.split("'s")
    inputArray = [x.strip() for x in inputArray]
    relation_calculate_func_map = {'father': (1, 'm'),
                                   'parent': (1, ''),
                                   'mother': (1, 'f'),
                                   'brother': (0, 'm'),
                                   'sibling': (0, ''),
                                   'sister': (0, 'f'),
                                   'son': (-1, 'm'),
                                   'child': (-1, ''),
                                   'daughter': (-1, 'f')}
    relation_spouse_func_map = {'husband': ('m'),
                                'wife': ('f'),
                                'spouse': (' ')}

    for i in range(len(inputArray)):  # note we are skipping last array element
        clean_answer = myanswer.replace("(non-blood relative)", "").replace("(blood relative)", "").strip()
        if clean_answer and \
                clean_answer in ["daughter in law", "son in law", "brother in law", "sister in law", "siblings in law"]:
            # 因为这个计算器不支持从in law继续推理，所以这里将in law的关系转换成直接的关系
            Clear()
            map_dict = {'daughter in law': 'daughter',
                        'son in law': 'son',
                        'brother in law': 'brother',
                        'sister in law': 'sister',
                        'siblings in law': 'sibling'}
            kinship_calculate(map_dict[clean_answer])

        relation = inputArray[i]
        if relation in relation_calculate_func_map:
            Calculate(*relation_calculate_func_map[relation])
        elif relation in relation_spouse_func_map:
            Spouse(*relation_spouse_func_map[relation])
        elif relation == "uncle":
            kinship_calculate("father's brother")
            # Calculate(*relation_calculate_func_map["father"])
            # Calculate(*relation_calculate_func_map["brother"])
        elif relation == "aunt":
            kinship_calculate("father's sister")
            # Calculate(*relation_calculate_func_map["father"])
            # Calculate(*relation_calculate_func_map["sister"])
        elif relation == "nephew":
            kinship_calculate("brother's son")
            # Calculate(*relation_calculate_func_map["son"])
            # Calculate(*relation_calculate_func_map["brother"])
        elif relation == "niece":
            kinship_calculate("brother's daughter")
            # Calculate(*relation_calculate_func_map["brother"])
            # Calculate(*relation_calculate_func_map["daughter"])
        elif relation == "cousin":
            kinship_calculate("uncle's child")
        elif relation == "grandfather":
            kinship_calculate("father's father")
        elif relation == "grandmother":
            kinship_calculate("father's mother")
        elif relation == "grandson":
            kinship_calculate("son's son")
        elif relation == "granddaughter":
            kinship_calculate("son's daughter")
        elif relation == "grandparent":
            kinship_calculate("father's parent")
        elif relation == "grandchild":
            kinship_calculate("son's child")
        elif relation == "son-in-law" or relation == "son in law":
            kinship_calculate("daughter's husband")
        elif relation == "daughter-in-law" or relation == "daughter in law":
            kinship_calculate("son's wife")
        # elif relation == "parent-in-law" or relation == "parent in law": # 这个模拟不出来，也碰不到。father's wife或
        # father's spouse这种，可以生成step mother
        #     kinship_calculate("child's parent")
        elif relation == "grand-nephew" or relation == "grand nephew" or relation == "great nephew":
            kinship_calculate("brother's grandson")
        elif relation == "grand-niece" or relation == "grand niece" or relation == "great niece":
            kinship_calculate("brother's granddaughter")
        elif relation == "sister-in-law" or relation == "sister in law":
            kinship_calculate("brother's wife")
        elif relation == "brother-in-law" or relation == "brother in law":
            kinship_calculate("sister's husband")
        elif relation == "sibling-in-law" or relation == "sibling in law":
            kinship_calculate("sibling's spouse")
        elif relation == "great-grandfather" or relation == "great grandfather":
            kinship_calculate("father's father's father")
        elif relation == "great-grandmother" or relation == "great grandmother":
            kinship_calculate("father's father's mother")
        elif relation == "great-grandson" or relation == "great grandson":
            kinship_calculate("son's son's son")
        elif relation == "great-granddaughter" or relation == "great granddaughter":
            kinship_calculate("son's son's daughter")
        elif relation == "great-grandparent" or relation == "great grandparent":
            kinship_calculate("father's father's parent")
        elif relation == "great-grandchild" or relation == "great grandchild":
            kinship_calculate("son's son's child")
        elif relation == "step-daughter" or relation == "step daughter":#这个好像模拟不出来，就先用daughter了
            kinship_calculate("daughter")
        elif relation == "step-son" or relation == "step son":
            kinship_calculate("son")
        elif relation == "step-parent" or relation == "step parent":
            kinship_calculate("parent's spouse")
        elif relation == "step-father" or relation == "step father":
            kinship_calculate("father's wife")
        elif relation == "step-mother" or relation == "step mother":
            kinship_calculate("mother's husband")
        elif relation == "step-sibling" or relation == "step sibling":
            kinship_calculate("sibling's spouse")
        elif relation == "step-brother" or relation == "step brother":
            kinship_calculate("step-parent's son")
        elif relation == "step-sister" or relation == "step sister":
            kinship_calculate("step-parent's daughter")
        elif relation == "step-grandparent" or relation == "step grandparent":
            kinship_calculate("step-parent's parent")
        elif relation == "step-grandfather" or relation == "step grandfather":
            kinship_calculate("step-parent's father")
        elif relation == "step-grandmother" or relation == "step grandmother":
            kinship_calculate("step-parent's mother")
        elif relation == "step-uncle" or relation == "step uncle":
            kinship_calculate("step-parent's brother")
        elif relation == "step-aunt" or relation == "step aunt":
            kinship_calculate("step-parent's sister")
        elif relation == "step-daughter" or relation == "step daughter": # 这个好像模拟不出来，就先用daughter了
            kinship_calculate("daughter")
        elif relation == "step-son" or relation == "step son":
            kinship_calculate("son")
        elif relation == "step-child" or relation == "step child":
            kinship_calculate("child")
        else:
            raise Exception("invalid relation: " + relation)

        Display()

def main(input_text):
    global myanswer, myanswer2, input1
    Clear()
    kinship_calculate(input_text.strip().lower())
    Display()
    print(input1)
    print(myanswer)
    print(myanswer2)

    myanswer = myanswer.replace("(blood relative)", "").replace("(non-blood relative)", "").strip()

    return myanswer

# main("brother's brother")

def kinship_calculator(input_text: str):
    """
    sister'sister is sister
    """
    pattern = re.compile(r"(.*)'s (.*) is (.*)")
    match = pattern.match(input_text)

    def clean_relation(text):
        """将亲属关系的文本简化"""
        text = text.replace('-', ' ').replace(' ', '').replace('inlaw', '').replace("great", "grand").replace("still",
                                                                                                              "")
        text = text.replace("(bloodrelative)", "").replace("(non-bloodrelative)", "").replace("step", "").strip()
        if text == "sibling":
            text = ["brother", "sister", "cousin", "sibling"]
        elif text == "1stcousinonceremoved":
            text = ["niece", "nephew"]
        elif text == "1stcousintwiceremoved":
            text = ["grandniece", "grandnephew"]

        text = text.replace("1st", "").replace("2nd", "").replace("3rd", "")

        return text

    try:
        if match:
            input_text = match.group(1) + "'s " + match.group(2)
            label = clean_relation(match.group(3))
            pred = clean_relation(main(input_text))
            if label == pred:
                return True
            elif {label, pred} in [{'aunt', 'mother'}, {'uncle', 'father'}]:
                return True
            elif isinstance(pred, list) and label in pred:
                return True
                # label in ["niece", "nephew"] and "1stcousinonceremoved" in pred:
                # print("correct")
            # elif label == "sibling" and ("brother" in pred or "sister" in pred or "cousin" in pred):
            #     print("correct")
            # elif label in ["brother", "sister", "cousin"] and "sibling" in pred:
            #     print("correct")
            else:
                print("错误的例子：", input_text, "预测答案：", label, "正确答案：", pred)
                return False

        else:
            print("无法解析的例子：", input_text)
            return False

    except Exception as e:
        print(e)
        print("格式不正确的输入：", input_text)
        return False


# if __name__ == '__main__':
#     from score import sample_rule_base
#     import re
#
#     # sample_rule_base = ["mother's son is son", "son's grandmother is grandmother", "son's brother is brother",
#     #                            "son's sister is sister", "brother's mother is grandmother", "son's uncle is father",
#     #                            "nephew's sister is sister", "aunt's sister is sister", "mother's son is nephew",
#     #                            "nephew's brother is uncle", "grandfather's sister is sister", "father's daughter is daughter",
#     #                            "daughter's brother is brother", "father's brother is brother", "daughter's brother is brother",
#     #                            "daughter's sister is sister", "son's aunt is aunt", "cousin's son is cousin", "cousin's brother is nephew",
#     #                            "daughter's mother is aunt", "cousin's brother is nephew", "nephew's uncle is uncle",
#     #                            "nephew's daughter is niece", "nephew's son is son", "sister's mother is daughter",
#     #                            "daughter's son is son", "son's mother is mother", "daughter's father is father",
#     #                            "father's daughter is niece", "daughter's brother is sibling", "mother's daughter is daughter",
#     #                            "daughter's mother is mother", "nephew's aunt is aunt", "cousin's son is great nephew",
#     #                            "great nephew's brother is great uncle",
#     #                            "father's son is son", "nephew's sister is also sister"]
#
#     for input_text in sample_rule_base:
#         #Aunt's sister is aunt
#         pattern = re.compile(r"(.*)'s (.*) is (.*)")
#         match = pattern.match(input_text)
#
#         def clean_relation(text):
#             """将亲属关系的文本简化"""
#             text = text.replace('-', ' ').replace(' ', '').replace('inlaw', '').replace("great", "grand").replace("still", "")
#             text = text.replace("(bloodrelative)", "").replace("(non-bloodrelative)", "").replace("step", "").strip()
#             if text == "sibling":
#                 text = ["brother", "sister", "cousin", "sibling"]
#             elif text == "1stcousinonceremoved":
#                 text = ["niece", "nephew"]
#             elif text == "1stcousintwiceremoved":
#                 text = ["grandniece", "grandnephew"]
#
#             text = text.replace("1st", "").replace("2nd", "").replace("3rd", "")
#
#             return text
#
#
#         try:
#             if match:
#                 input_text = match.group(1) + "'s " + match.group(2)
#                 label = clean_relation(match.group(3))
#                 pred = clean_relation(main(input_text))
#                 if label == pred:
#                     print("correct")
#                 elif {label, pred} in [{'aunt', 'mother'}, {'uncle', 'father'}]:
#                     print("correct")
#                 elif isinstance(pred, list) and label in pred:
#                     print("correct")
#                         # label in ["niece", "nephew"] and "1stcousinonceremoved" in pred:
#                     # print("correct")
#                 # elif label == "sibling" and ("brother" in pred or "sister" in pred or "cousin" in pred):
#                 #     print("correct")
#                 # elif label in ["brother", "sister", "cousin"] and "sibling" in pred:
#                 #     print("correct")
#                 else:
#                     print("错误的例子：", input_text, "正确答案：", label, "预测答案：", pred)
#
#             else:
#                 print("无法解析的例子：", input_text)
#
#         except Exception as e:
#             print(e)
#             print("格式不正确的输入：", input_text)