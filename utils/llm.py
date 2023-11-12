from typing import Optional, List
from utils.llm_models.call_openai import call_openai

generate_func_mapping = {"davinci": call_openai,
                         "gpt-3.5-turbo": call_openai,
                         "gpt-3.5-turbo-0613": call_openai}


class LLM:
    def __init__(self, generate_single):
        self.generate_single = generate_single

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate_single(prompt, **kwargs) for prompt in prompts]

if __name__ == '__main__':
    llm = LLM(call_openai)
    print(llm.generate_single('hello world'))