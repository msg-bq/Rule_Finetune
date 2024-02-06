import re

from utils.ExtraNameSpace import PredictionCleanNameSpace
import string


@PredictionCleanNameSpace.register("Default")
def clean_prediction(self, prediction: str) -> str:
    return prediction

@PredictionCleanNameSpace.register("CLUTRR")
def clean_prediction(self, prediction: str) -> str:
    """
    从Answer中提出特定数据集的答案
    这里传入的pred已经是最后一个The answer is后面的部分了
    """
    if prediction == "":
        return prediction

    pred_words = prediction.split()
    if len(pred_words) == 1:
        if pred_words[0][-1] in string.punctuation:
            return pred_words[0][:-1]

        return pred_words[0].strip()

    tags = ['<Begin>', '</End>', '<rule>', '<retrieved_rule>', '<new_rule>']
    for tag in tags:
        if tag in prediction:
            return clean_prediction(self, prediction.split(tag)[0])

    if pred_words[-1][-1] in string.punctuation:
        return pred_words[-1][:-1]

    return pred_words[-1].strip()

@PredictionCleanNameSpace.register("STS_B")
def clean_prediction(self, prediction: str) -> str:
    prediction = prediction.strip().lower()
    result = ""

    pattern = "The sentiment of the above review is (.*)"
    match = re.match(pattern, prediction)
    if match:
        result = match.group(1)

    if not result:
        words = prediction.split()
        if len(words) == 1:
            result = prediction
        else:
            result = words[-1]


    if result[-1] in string.punctuation:
        return result[:-1]

    return result.strip()


@PredictionCleanNameSpace.register("LANG_8")
def clean_prediction(self, prediction: str) -> str:
    """
    GOLD_LABEL这个特殊的返回值表明答案是正确的
    """
    prediction = prediction.strip().lower()

    no_error_trigger = ["no grammar errors", "no grammatical errors"]
    if any(trigger in prediction for trigger in no_error_trigger):
        return "GOLD_LABEL"

    def remove_quotation_marks(text):
        """去除首尾的引号"""
        text = text.strip()
        if not text:
            return text

        if text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        elif text[0] == "“" and text[-1] == "”":
            return text[1:-1]
        elif text[0] == "'" and text[-1] == "'":
            return text[1:-1]
        return text

    pattern1 = "the revised sentence can be:(.*)"
    pattern2 = "the revised sentence could be \"(.*)\""
    pattern3 = "the revised sentence is:(.*)"
    pattern4 = "the revised sentence would be:(.*)"
    pattern5 = "it could be revised as follows:(.*)"
    pattern6 = "however, a revised version of the sentence could be:(.*)"
    pattern7 = "revised sentence:(.*)"
    pattern8 = "the revised sentence should be:(.*)"
    pattern9 = "the correct sentence is:(.*)" # 最好是用""把每个(.*)包起来，不过有少数确实没有引号

    pattern_list = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9]
    for pattern in pattern_list:
        # 以\n或结束符结束
        pattern += "(\n|$)"
        match = re.search(pattern, prediction)
        if match:
            result = remove_quotation_marks(match.group(1))
            if result:
                return result

    return prediction

@PredictionCleanNameSpace.register("HtT_version")
def clean_prediction(self, prediction: str) -> str:
    pattern = "therefore, the answer is (.*)"
    match = re.search(pattern, prediction.lower())
    if match:
        return match.group(1)

    return prediction