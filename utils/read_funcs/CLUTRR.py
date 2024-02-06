# import os
# from typing import List
# import random
# import pandas as pd
#
#
# question_template = '''Context: {story}\n'''\
# '''Question: {query[1]} is {query[0]}'s what?\n'''
#
# def read_CLUTRR_data(path):
#     train_task = '1.2,1.3'  # 有不同的拆分方案，不过我们只拿这个做实验
#     test_task = '1.4,1.5,1.6,1.7,1.8,1.9,1.10' #1.2,1.3,
#
#     train_file = f'{train_task}_train.csv'
#     test_files = [f'{task.strip()}_test.csv' for task in test_task.split(',')]
#
#     train_data = pd.read_csv(os.path.join(path, train_file))
#     test_data = [pd.read_csv(os.path.join(path, test_file)) for test_file in test_files]
#
#     return train_data, test_data
#
#
# def build_samples(original_datasets: List[dict]):
#     """
#     这里是将原始数据转换成字典格式、且只选取了必要的key
#     """
#     for i in range(len(original_datasets)):
#         if type(original_datasets[i]['query']) == str:
#             original_datasets[i]['query'] = eval(original_datasets[i]['query'])
#
#     return original_datasets
#
#
# def build_datasets_from_samples(sampling_datasets: List[dict], question_template: str = question_template) -> List[dict]:
#     """
#     这里是将已经转成字典格式、且只选取了必要的key的数据。我们将其转换成符合CoT格式的数据
#     """
#     final_datasets = []
#     for i in range(len(sampling_datasets)):
#         sampling_datasets[i]['story'] = sampling_datasets[i]['story'].replace('[', '').replace(']', '')
#         query = question_template.format(**sampling_datasets[i])
#         target = sampling_datasets[i]['target']
#
#         final_datasets.append({'question': query, 'gold_label': target})
#
#     return final_datasets
#
#
# def read_CLUTRR(data_dir):
#     train_data, test_data = read_CLUTRR_data(data_dir)
#     keys = ['query', 'story', 'target']
#     train_data = train_data[keys]
#     sampling_train_data = build_samples(train_data.to_dict(orient='records'))
#     final_train_datasets = build_datasets_from_samples(sampling_train_data, question_template)[:200]
#     # final_train_datasets = random.sample(final_train_datasets, 1000)
#
#     test_data = [test_data[i][keys] for i in range(len(test_data))]
#     proportional_sampling = [31, 25, 16, 21, 30, 26, 25]  # 每个测试集所抽出的数量，1.2 1.3对应8, 18
#     sampling_test_data = []
#     for i in range(len(test_data)):
#         samples = test_data[i].sample(proportional_sampling[i], random_state=42)
#         samples = samples.to_dict(orient='records')
#         sampling_test_data.extend(samples)
#
#     sampling_test_data = build_samples(sampling_test_data)
#     final_test_datasets = build_datasets_from_samples(sampling_test_data, question_template)
#
#     return final_train_datasets, None, final_test_datasets


# ===============================
import os
from typing import List
from utils.ExtraNameSpace import DatasetsReaderNameSpace

import pandas as pd


question_template = '''Context: The relations on the path from {query[0]} to {query[1]} are {relation_path_2}.\n'''\
'''Question: {query[1]} is {query[0]}'s what?\n'''

def read_CLUTRR_data(path):
    train_task = '1.2,1.3'  # 有不同的拆分方案，不过我们只拿这个做实验
    test_task = '1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10' #1.2,1.3,

    train_file = f'{train_task}_train.csv'
    test_files = [f'{task.strip()}_test.csv' for task in test_task.split(',')]

    train_data = pd.read_csv(os.path.join(path, train_file))
    test_data = [pd.read_csv(os.path.join(path, test_file)) for test_file in test_files]

    return train_data, test_data


def build_samples(original_datasets: List[dict]):
    """
    这里是将原始数据转换成字典格式、且只选取了必要的key
    """
    for i in range(len(original_datasets)):
        if type(original_datasets[i]['query']) == str:
            original_datasets[i]['query'] = eval(original_datasets[i]['query'])

        if type(original_datasets[i]['edge_types']) == str:
            original_datasets[i]['edge_types'] = eval(original_datasets[i]['edge_types'])

    return original_datasets


def build_datasets_from_samples(sampling_datasets: List[dict], question_template: str = question_template) -> List[dict]:
    """
    这里是将已经转成字典格式、且只选取了必要的key的数据。我们将其转换成符合CoT格式的数据
    """
    final_datasets = []
    for i in range(len(sampling_datasets)):
        edge_type = sampling_datasets[i]['edge_types']
        relation_path_2 = ",".join(edge_type)
        sampling_datasets[i]['relation_path_2'] = relation_path_2
        query = question_template.format(**sampling_datasets[i])
        target = sampling_datasets[i]['target']

        final_datasets.append({'question': query, 'gold_label': target})

    return final_datasets

def remove_duplicates(data):
    """
    删除数据集的重复部分，并保持原有的顺序
    """
    seen = set()
    new_data = []
    for d in data:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_data.append(d)
    return new_data

@DatasetsReaderNameSpace.register("CLUTRR")
def read_func(data_dir):
    train_data, test_data = read_CLUTRR_data(data_dir)
    keys = ['query', 'edge_types', 'target']
    train_data = train_data[keys]
    sampling_train_data = build_samples(train_data.to_dict(orient='records'))
    final_train_datasets = build_datasets_from_samples(sampling_train_data, question_template)

    test_data = [test_data[i][keys] for i in range(len(test_data))]
    proportional_sampling = [8, 18, 31, 25, 16, 21, 30, 26, 25]  # 每个测试集所抽出的数量
    sampling_test_data = []

    for i in range(len(test_data)):
        samples = test_data[i].sample(proportional_sampling[i], random_state=42)
        samples = samples.to_dict(orient='records')
        sampling_test_data.extend(samples)

    sampling_test_data = build_samples(sampling_test_data)
    final_test_datasets = build_datasets_from_samples(sampling_test_data, question_template)

    final_train_datasets = remove_duplicates(final_train_datasets)
    final_test_datasets = remove_duplicates(final_test_datasets)

    return final_train_datasets, None, final_test_datasets
