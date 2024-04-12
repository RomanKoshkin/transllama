import os, sys
from dotenv import load_dotenv
from tqdm import tqdm
import os, json, nltk, re
from termcolor import cprint
from typing import List
from transformers import LlamaTokenizerFast
from datasets import Dataset
import numpy as np
from .constants import B_INST, E_INST, DEFAULT_SYSTEM_PROMPT, B_SYS, E_SYS
from nltk.tokenize import TreebankWordDetokenizer
from janome.tokenizer import Tokenizer


def expandFillers(tokenizer: LlamaTokenizerFast, src_words: List[str], tgt_words: List[str]):
    """
    Args:
        src_words, tgt_words have exactly the same number of nltk tokens,
        tokens in src_words are causal to tgt_words because of the fillers "@"
        This causality may be broken after tokenizing the sentences with the sentencepiece tokenizer
        because multitoken source words will turn out to be longer than the single fillers (@).
        To fix that, this function replaces the single fillers with multiple to match the 
        number of sentencepiece tokens in the source.
    Returns:
        tgt_words in which the fillers have been expanded 
    """

    for i, (s, t) in enumerate(zip(src_words, tgt_words)):
        if t == "@":
            num_src_subwrd_tokens = len(tokenizer(s, add_special_tokens=False)['input_ids'])
            tgt_replacement = " ".join(["@" for _ in range(num_src_subwrd_tokens)])
            tgt_words[i] = tgt_replacement

    return tgt_words


def makeCLMdataset(tokenizer, PATH_TO_ALIGNED):
    """
    MAKE DATA FOR AN DECODER-ONLY TRANSFORMER (llama-chat family)
    """

    S, T = [], []
    for fname in tqdm(os.listdir(PATH_TO_ALIGNED)):
        if not fname.endswith('.json'):
            continue
        # load a list of dicts. Each dict is a mapping from source to target sentence nltk words
        with open(f'{PATH_TO_ALIGNED}/{fname}', 'r') as f:
            d = json.loads(f.read())

        src_words = [it['source'] for it in d]
        tgt_words = [it['target'] for it in d]

        # sanity check
        try:
            assert len(src_words) == len(
                tgt_words
            ), f"The number of source and target words don't match. Check {PATH_TO_ALIGNED}/{fname}. Probably poorly aligned. "
        except Exception as e:
            cprint(e, color='red')

        # expand the number of target fillers (from 1 to N, where N is the number of sentencepiece tokens in the source word)
        t = expandFillers(tokenizer, src_words, tgt_words)

        # merge the list of nltk words to whole sentences, removing extra spaces around punctuation
        src_words = ' '.join(src_words)
        src_words = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', src_words)  # fix the punctuation
        tgt_words = ' '.join(t)
        tgt_words = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', tgt_words)  # fix the punctuation
        S.append(src_words)
        T.append(tgt_words)

    return Dataset.from_dict({"src": S, "tgt": T})


class EuTokenizer:

    def __init__(self):
        self.Detokenize = TreebankWordDetokenizer().detokenize
        self.Tokenize = nltk.word_tokenize

    def replace_with_unk(self, lst: list) -> list:
        return [i.replace("▁▁", "▁▁").strip() for i in lst]  # FIXME !!!!!!!!!!!!!!

    def tokenize(self, sentence):
        return self.replace_with_unk(self.Tokenize(sentence))

    def detokenize(self, x):
        return self.Detokenize(x)

    def get_full_words(self, sentence):
        tokens = self.Tokenize(sentence)
        if len(tokens) == 0:
            return tokens
        if tokens[-1] == '.':
            tokens[-2] += '.'
            return tokens[:-1]
        return tokens


class JaTokenizer:

    def __init__(self):
        self.tokenizer = Tokenizer()

    def fix_lumped_wait_tokens(self, lst: list) -> list:
        result = []
        for item in lst:
            if "▁▁" in item:
                result.extend([item[i:i + 2] for i in range(0, len(item), 2)])
            else:
                result.append(item)
        return result

    def replace_with_unk(self, lst: list) -> list:
        result = []
        for item in lst:
            if "▁▁" in item:
                result.extend([" ▁▁" for i in range(0, len(item), 2)])
            else:
                result.append(item)
        return result

    def tokenize_whole_sentence(self, sentence):
        tokens = self.replace_with_unk([token.surface for token in self.tokenizer.tokenize(sentence)])
        if "。" not in tokens[-1]:
            tokens[-1] += "。"
        return tokens

    def tokenize(self, sentence):
        return self.replace_with_unk([token.surface for token in self.tokenizer.tokenize(sentence)])

        # return self.fix_lumped_wait_tokens([token.surface for token in self.tokenizer.tokenize(sentence)])

    def detokenize(self, x):
        return ''.join(x)

    def get_full_words(self, sentence):
        tokens = self.tokenize(sentence)
        if len(tokens) < 1:
            return []
        elif tokens[-1] == '。':
            tokens[-2] += '。'
            return tokens[:-1]
        elif tokens[-1].endswith('。'):
            return tokens
        else:  # if not the end and more than two words
            return tokens[:-1]


def makeCLMdatasetFromListOfDictsOfAlignedWords(tokenizer, D, japanese_target=False):
    """
    MAKE DATA FOR AN DECODER-ONLY TRANSFORMER (llama-chat family)
    """
    detokenizer = TreebankWordDetokenizer()
    if japanese_target:
        ja_tokenizer = JaTokenizer()

    S, T = [], []
    for i, d in enumerate(tqdm(D)):

        src_words = [w for w in d['source']]
        tgt_words = [w for w in d['target']]

        # sanity check
        try:
            assert len(src_words) == len(
                tgt_words
            ), f"The number of source and target words don't match. Check {i}th item. Probably poorly aligned. "
        except Exception as e:
            cprint(e, color='red')
            continue

        # expand the number of target fillers (from 1 to N, where N is the number of sentencepiece tokens in the source word)
        # t = expandFillers(tokenizer, src_words, tgt_words)

        # merge the list of nltk words to whole sentences, removing extra spaces around punctuation
        # src_words = ' '.join(src_words) # NOTE: this leaves unwanted spaces in isn't, don't etc
        src_words = detokenizer.detokenize(src_words)
        src_words = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', src_words)  # fix the punctuation
        # tgt_words = ' '.join(t)
        tgt_words = detokenizer.detokenize(tgt_words) if not japanese_target else ja_tokenizer.detokenize(tgt_words)
        tgt_words = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', tgt_words)  # fix the punctuation
        S.append(src_words)
        T.append(tgt_words)

    return Dataset.from_dict({"src": S, "tgt": T})


# make a dataset that returns a dict of {src_str, tgt_str}
def preprocess_and_sample(tokenizer, data_point, sub_sample_prefix=True):

    src_sent = data_point['src'].strip()
    tgt_sent = data_point['tgt'].strip()

    if sub_sample_prefix:
        src_words = nltk.word_tokenize(src_sent)
        tgt_words = nltk.word_tokenize(tgt_sent)
        full_length = len(src_words)
        sample_length = np.random.randint(full_length)  # NOTE: maybe sample longer sequences more.

        src_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', ' '.join(src_words[:sample_length]))
        tgt_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', ' '.join(tgt_words[:sample_length]))
        cprint(src_sent, color='red')
        cprint(tgt_sent, color='blue')

        # only add EOS if the tranlation if finished
        if sample_length == full_length:
            string = (f"{tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent} {tokenizer.eos_token}")
        else:
            string = (f"{tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent}")
    src = tokenizer(string, add_special_tokens=False)

    return {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}


if __name__ == "__main__":
    pass
    # make a dataset that returns a dict of {token_str, tgt_str}
    # data = ds.shuffle().map(preprocess_and_sample)
    # data = ds.map(preprocess_and_sample)