from typing import List

from utils.ExtraNameSpace import DatasetsReaderNameSpace

question_template = '''Sentence: {}\n'''\
'''Instruction: Correct the grammar errors in the sentence and return the corrected sentence.\n'''

question_template = '''Sentence: {}\n'''\
'''Question: What's the grammar errors and revised sentence of above sentence? '''

def read_LANG_8_data(path) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        if lines[i] == '':
            continue
        lines_data = lines[i].split('\t')
        if len(lines_data) == 5:
            label = lines_data[4]
        else:
            label = lines_data[5]
        sentence = lines_data[4]
        data.append({'sentence': sentence, 'label': label})
    data = [{'question': question_template.format(sample['sentence']), 'gold_label': sample['label']}
            for sample in data]
    return data


@DatasetsReaderNameSpace.register("LANG_8")
def read_func(data_dir):
    train_data = read_LANG_8_data(f'{data_dir}/lang-8-en-1.0/entries.train')[:200]
    test_data = read_LANG_8_data(f'{data_dir}/lang-8-en-1.0/entries.test')[:200]

    return train_data, None, test_data

