import openai, tiktoken
import string, warnings
# nltk.download('cmudict')
from nltk.corpus import cmudict
import pandas as pd
import re

d = cmudict.dict()


def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        return 0


def cntSylInEngTxt(txt):
    A = 0
    txt = txt.replace('-', ' ')
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    for word in txt.split(' '):
        A += nsyl(word)
        # print(word, nsyl(word), A)
    return A


def get_initial_prompt(prefix, prompt_name="prompt_template_1a"):
    with open(f'../data/prompt_templates/{prompt_name}.txt', 'r') as f:
        template = f.read()
    prompt = template.replace(f'{{prefix}}', prefix)
    return prompt


def get_choice_prompt(prompt_name="prompt_template_1a", accumulated_concise_translation=""):
    with open(f'../data/prompt_templates/{prompt_name}.txt', 'r') as f:
        prompt = f.read()
    if len(accumulated_concise_translation) == 0:
        return prompt
    else:
        prompt = f'{prompt} Make sure the concise version is also a good fit (syntactically and grammatically) with the previous translation: "{accumulated_concise_translation}.'
        return prompt


def get_followup_prompt(prefix, accumulated_concise_translation, prompt_name="prompt_template_2a"):
    with open(f'../data/prompt_templates/{prompt_name}.txt', 'r') as f:
        template = f.read()
    prompt = template.replace(f'{{prefix}}', prefix)
    prompt = prompt.replace(f'{{accumulated_concise_translation}}', accumulated_concise_translation)
    return prompt


def call_api(prompt, model_name, temperature=0.5, max_tokens=10000, extra_msg=[], sys_prompt_name='system_prompt_1'):
    with open(f'../data/prompt_templates/{sys_prompt_name}.txt', 'r') as f:
        system_prompt = f.read()
    messages = [{
        "role": "system",
        "content": system_prompt,
    }, {
        "role": "user",
        "content": prompt,
    }]

    if len(extra_msg) > 0:
        for msg in extra_msg:
            messages.append(msg)
    # print(messages)
    r = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r


class Chunks:

    def __init__(self, chunked_text):
        self.chunked_text = chunked_text

    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise NotImplementedError("Slicing this object is not implemented yet.")
            start = i.start
            stop = i.stop
            return MyIterable(self.data[index])
        else:
            return " ".join(self.chunked_text[:i])

    def __next__(self):
        self.i += 1
        if self.i <= self.__len__():
            return " ".join(self.chunked_text[:self.i])
        else:
            raise StopIteration

    def __len__(self):
        return len(self.chunked_text)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def log_usage_stats(r, MAX_TOKENS, MODEL_NAME):
    total_sent_tokens = r.usage.prompt_tokens
    total_received_tokens = r.usage.completion_tokens
    if total_received_tokens >= MAX_TOKENS:
        warnings.warn('MAX_TOKENS exceeded')
    with open(f'../data/logs/{MODEL_NAME}_token_counts.log', 'a') as f:
        f.write(
            f"unix_s,{int(pd.Timestamp.now().timestamp())},sent,{total_sent_tokens},received,{total_received_tokens}\n")


def make_prompt(verbose_source, PROMPT_TEMPLATE_NAME):
    with open(f'../data/prompt_templates/{PROMPT_TEMPLATE_NAME}.txt', 'r') as f:
        prompt = re.sub("\{verbose_source\}", verbose_source, f.read())
    return prompt