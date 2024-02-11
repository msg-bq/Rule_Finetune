import re

from LANG_8 import read_LANG_8_data
from utils.llm_models.call_openai import call_openai

dataset = read_LANG_8_data(r"D:\Downloads\clang8-main\output_data\clang8_source_target_en.spacy_tokenized.tsv")
test_dataset = dataset[-200:]

save_path = r"D:\Downloads\clang8-main\output_data\test_gpt4"
clean_clang8_path = r"D:\Downloads\clang8-main\output_data\clean_clang8"
with open(save_path, 'w', encoding='utf-8') as f:
    pass
with open(clean_clang8_path, 'w', encoding='utf-8') as f:
    pass

clean_clang8 = []
gpt4_response = []

def clean_response(response):
    response = response.strip()
    return response

for sample in test_dataset:
    pattern = "Sentence: (.*)\nQuestion: What's the grammar errors and revised sentence of above sentence?"
    original_sentence = re.match(pattern, sample['question']).group(1).strip()

    prompt = f"Please correct the grammatical error:\n{original_sentence}\nJust output the modified sentence."
    response = call_openai(prompt, model="gpt-4-1106-preview")
    response = (original_sentence, response)
    # gpt4_response.append((original_sentence, response))

    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(str(response) + '\n')

    with open(clean_clang8_path, 'a', encoding='utf-8') as f:
        response = (response[0], clean_response(response[1]))
        f.write(str(response[0])+'\t' + str(response[1]) + '\n')

# if __name__ == '__main__':
# prompt = "Please correct the grammatical error: \n" \
#          "Let 's imagine how much money they can save without smoking . \n" \
#          "Just output the modified sentence."
#
# print(call_openai(prompt, model="gpt-4-1106-preview"))