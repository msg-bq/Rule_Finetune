from utils.ExtraNameSpace import ColdStartScoreNameSpace


@ColdStartScoreNameSpace.register("Default")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    """
    用于在cold_start阶段判断是否应当保留prediction对应的rationale
    """
    return prediction.strip() == gold_label.strip()


@ColdStartScoreNameSpace.register("LANG_8")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    cleaned_pred = prediction.strip().replace(' ', '').lower()
    cleaned_gold = gold_label.strip().replace(' ', '').lower()
    if cleaned_gold in cleaned_pred:
        return True

    return False