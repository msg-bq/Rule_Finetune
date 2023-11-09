import os
from read_funcs.CLUTRR import read_CLUTRR
from data import Example, DatasetLoader
from operator import itemgetter

"""
为不同的数据集，准备不同的读入方式，输入都是一个dir，输出train, dev, test, 输出统一{'question', 'gold_label'}
最终使用的数据集，将被单独保存为train_preprocessed.jsonl, dev_preprocessed.jsonl, test_preprocessed.jsonl
"""

read_func_mapping = {'CLUTRR': read_CLUTRR}


def read_preprocessed_data(path):
    with open(path, 'r') as f:
        data = [eval(line.strip()) for line in f.readlines()]
    return data


def save_preprocessed_data(data, path):
    with open(path, 'w') as f:
        for sample in data:
            f.write(str(sample) + '\n')


def adjust_dataset_format(**kwags):
    """
    将read_datasets输出的内容，调整为DatasetLoader的格式
    """
    for key, value in kwags.items():
        if value:
            kwags[key] = DatasetLoader([Example(**sample) for sample in value])
    return kwags


def read_datasets(args):
    data_dir = args.data_dir

    # 存储在preprocessed_data下面
    train_path = os.path.join(data_dir, 'preprocessed_data/train_preprocessed.jsonl')
    test_path = os.path.join(data_dir, 'preprocessed_data/test_preprocessed.jsonl')
    valid_path = os.path.join(data_dir, 'preprocessed_data/valid_preprocessed.jsonl')

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_dataset = read_preprocessed_data(train_path)
        test_dataset = read_preprocessed_data(test_path)
        if os.path.exists(valid_path):
            valid_dataset = read_preprocessed_data(valid_path)
        else:
            valid_dataset = None

    else:
        read_func = read_func_mapping[args.dataset]
        train_dataset, valid_dataset, test_dataset = read_func(data_dir)

        save_preprocessed_data(train_dataset, train_path)
        save_preprocessed_data(test_dataset, test_path)
        if valid_dataset:
            save_preprocessed_data(valid_dataset, valid_path)

    train_dataset, valid_dataset, test_dataset = \
        itemgetter('train', 'valid', 'test')(adjust_dataset_format(train=train_dataset,
                                                                   valid=valid_dataset,
                                                                   test=test_dataset))

    return train_dataset, valid_dataset, test_dataset


def read_rationales(args):
    data_dir = args.rationale_dir

    train_path = os.path.join(data_dir, 'train.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')
    valid_path = os.path.join(data_dir, 'valid.jsonl')

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_dataset = read_preprocessed_data(train_path)
        test_dataset = read_preprocessed_data(test_path)
        if os.path.exists(valid_path):
            valid_dataset = read_preprocessed_data(valid_path)
        else:
            valid_dataset = None

    else:
        raise ValueError('rationale_dir is not valid')

    train_dataset, valid_dataset, test_dataset = \
        itemgetter('train', 'valid', 'test')(adjust_dataset_format(train=train_dataset,
                                                                   valid=valid_dataset,
                                                                   test=test_dataset))

    return train_dataset, valid_dataset, test_dataset