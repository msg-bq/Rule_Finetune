from random import random
from typing import Optional, List
import time
import openai
import tiktoken

openai.api_base = "https://api.chatanywhere.com.cn"
key_list = ["sk-0LsJe3iHrbu4HNOK01R7bDBloOlKdtHB92j8BuhS5uzrkM95"]
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


def call_openai(input_text: Optional[List[str], str], model="gpt-3.5-turbo-0613", is_gpt3=False, **kwargs) -> str:
    """
    lbq
    List[str] GPT存在历史，str 不存在历史
    """
    if 'topN' in kwargs:
        kwargs['n'] = kwargs.pop('topN')

    max_supported_tokens = 8000 if model.startswith("gpt-4") else 4000

    if isinstance(input_text, list):
        prompt = input_text[:max_supported_tokens]
    else:
        prompt = [{"role": "user", "content": input_text[:max_supported_tokens]}]


    if 'max_length' in kwargs:
        max_length = kwargs['max_length']
        max_length = min(max_length, max_supported_tokens - cnt_tokens(prompt))
        if max_length < 0:
            raise ValueError("max_length is too small")

    global key_choose
    openai.api_key = key_list[key_choose]

    while True:
        try:
            if is_gpt3:
                result = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    **kwargs)
                return result['choices'][0]['text']
            else:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    **kwargs
                )

                return completion.choices[0].message['content'].strip()

        except Exception as e:
            print(e)
            time.sleep(20 + 10 * random())
            key_choose = (key_choose + 1) % len(key_list)
