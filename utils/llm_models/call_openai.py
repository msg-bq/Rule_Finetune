import concurrent
from concurrent.futures import as_completed
from random import random
from typing import Optional, List, Union
import time
import openai
import tiktoken

openai.api_base = "https://api.chatanywhere.com.cn/v1"
key_list = ["sk-z3z3xHLy1H8zfVl7TPz8pFuWbGayeINXJ9alEvp5fQ7O0FVO"]
key_choose = 0

encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cnt_tokens(message):
    '''
    统计message的token数
    '''
    if isinstance(message, list):
        cnt = 0
        for m in message:
            cnt += num_tokens_from_string(m['content'])
    elif isinstance(message, str):
        cnt = num_tokens_from_string(message)
    else:
        raise TypeError("message should be list or str")

    return cnt


def call_openai(input_text: Union[List[str], str], model="gpt-3.5-turbo-1106", is_gpt3=False, **kwargs) \
        -> Union[str, List[str]]:
    """
    lbq
    List[str] GPT存在历史，str 不存在历史
    """
    if 'topN' in kwargs:
        kwargs['n'] = kwargs.pop('topN')

    max_supported_tokens = 6000 if model.startswith("gpt-4") else 3000 # ≈3:4

    if isinstance(input_text, str):
        prompt = [{"role": "user", "content": " ".join(input_text.split(' ')[:max_supported_tokens])}]
    else:
        prompt = input_text


    if 'max_length' in kwargs:
        max_length = kwargs['max_length']
        max_length = min(max_length, max_supported_tokens - cnt_tokens(prompt))
        if max_length < 0:
            raise ValueError("max_length is too small")

    global key_choose
    openai.api_key = key_list[key_choose]

    try_call = 10
    while try_call:
        try_call -= 1
        start_time = time.time()
        try:
            if is_gpt3:
                result = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    **kwargs)
                # print("time:", time.time() - start_time)
                if len(result.choices) == 1:
                    return result.choices[0].text.strip()
                else:
                    return [c.text.strip() for c in result.choices]

            else:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    **kwargs
                )
                # print("time:", time.time() - start_time)

                if len(completion.choices) == 1:
                    return completion.choices[0].message['content'].strip()
                else:
                    return [c.message['content'].strip() for c in completion.choices]

        except Exception as e:
            time.sleep(20 + 10 * random())
            key_choose = (key_choose + 1) % len(key_list)

    if 'n' in kwargs and kwargs['n'] == 1:
        return ''
    else:
        return []

if __name__ == '__main__':
    prompt = \
'''Context: The relations on the path from Michael to Alfred are brother, son.
Question: Alfred is Michael's what?
Answer: Let's think step by step. If you use some rules in the reasoning process, please write them in "<rule>xxx<rule>" format individually before you draw every conclusion.'''

#     prompt = \
#     """Context: Florence and her husband Norman went to go ice skating with their daughter Marilyn. Marilyn's sister Janet could not go because she has a broken leg. Kecia went to the store with her sister Florence Chris loved going to the store with his mom Florence. She always bought him snacks Chris likes to visit his sister. Her name is Janet.
# Question: Kecia is Norman's what?
# Answer: Let's think step byx step. If you use some rules in the reasoning process, please write them with "<rule>xxx<rule>" format individually."""

    print(prompt)
    num_worker = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        sub_response = [executor.submit(call_openai, prompt, model="gpt-4-1106-preview") for _ in range(num_worker)]

        for r in as_completed(sub_response):
            print(r.result())
    # print(call_openai(prompt, model="gpt-4"))