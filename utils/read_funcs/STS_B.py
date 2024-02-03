from typing import List

import pandas as pd

from utils.ExtraNameSpace import DatasetsReaderNameSpace

question_template = '''Review: {}\n'''\
'''Question: What's the sentiment of above review?\n'''

def read_STS_B_data(path) -> List[dict]:
    data = pd.read_csv(path, sep='\t', header=0)

    label2sentiment = {0: 'negative', 1: 'positive'}
    data['label'] = data['label'].apply(lambda x: label2sentiment[x])

    data = data.to_dict(orient='records')
    data = [{'question': question_template.format(sample['sentence']), 'gold_label': sample['label']}
            for sample in data]
    return data


@DatasetsReaderNameSpace.register("STS_B")
def read_func(data_dir):
    train_data = read_STS_B_data(f'{data_dir}/train.tsv')[:200]
    dev_data = read_STS_B_data(f'{data_dir}/dev.tsv')
    # test_data = read_STS_B_data(f'{data_dir}/test.tsv')

    return train_data, None, dev_data



