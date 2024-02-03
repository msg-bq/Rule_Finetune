from utils.ExtraNameSpace import ScoreNameSpace


@ScoreNameSpace.register("Default")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    """
    用于判断是否应当保留prediction对应的rationale
    """
    return prediction.strip() == gold_label.strip()


@ScoreNameSpace.register("LANG_8")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    cleaned_pred = prediction.strip().replace(' ', '').lower()
    cleaned_gold = gold_label.strip().replace(' ', '').lower()

    return cleaned_gold in cleaned_pred


@ScoreNameSpace.register("STS_B")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    prediction = prediction.lower().strip()
    gold_label = gold_label.lower().strip()

    return prediction.startswith(gold_label) or prediction.endswith(gold_label)