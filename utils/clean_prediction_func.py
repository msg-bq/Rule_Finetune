from utils.ExtraNameSpace import PredictionCleanNameSpace
import string


#不知道类内的那个@classmethod能不能生效，我理解可以
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

    tags = ['<B>', '<E>', '<rule>', '<retrieved_rule>', '<new_rule>']
    for tag in tags:
        if tag in prediction:
            return clean_prediction(self, prediction.split(tag)[0])

    if pred_words[-1][-1] in string.punctuation:
        return pred_words[-1][:-1]

    return pred_words[-1].strip()