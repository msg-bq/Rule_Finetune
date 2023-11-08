from random import random
from typing import Optional, List
import time
import openai

openai.api_base = "https://api.chatanywhere.com.cn"
key_list = ["sk-0LsJe3iHrbu4HNOK01R7bDBloOlKdtHB92j8BuhS5uzrkM95"]
key_choose = 0


def call_openai(input_text: Optional[List[str], str], model="gpt-3.5-turbo-0613", is_gpt3=False, **kwargs) -> str:
    """
    lbq
    List[str] GPT存在历史，str 不存在历史
    """
    if isinstance(input_text, list):
        prompt = input_text
    else:
        prompt = [{"role": "user", "content": input_text}]

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
