import os, sys
from tkinter import W
from dotenv import load_dotenv

from lib2to3.pytree import generate_matches
import torch
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from termcolor import cprint
import nltk, re
from nltk.tokenize import TreebankWordDetokenizer
from .constants import B_INST, E_INST, DEFAULT_SYSTEM_PROMPT, B_SYS, E_SYS, WAIT_TOKEN
from .datasets import JaTokenizer, EuTokenizer
from .llm_utils import fix_spaces_around_punctuation, remove_multiple_spaces


def get_num_words_wo_fillers(src_words: list, WAIT_TOKEN: str) -> int:
    i = 0
    for w in src_words:
        if w not in ["▁▁", WAIT_TOKEN]:
            i += 1
    return i


def drop_tail_fillers(string):
    return string.replace(WAIT_TOKEN, "").strip()


def dropAllWaitTokens(string, WAIT_TOKEN):
    # cprint(string, color='red')
    # cprint(string.replace(WAIT_TOKEN, '').strip(), color='green')
    return string.replace(WAIT_TOKEN, '').strip()


def leave_only_last_wait_token(string, WAIT_TOKEN):
    # cprint(string, color='red')

    if string.endswith(WAIT_TOKEN):
        # cprint((string.replace(WAIT_TOKEN, '') + WAIT_TOKEN).strip(), color='green')
        return (string.replace(WAIT_TOKEN, '') + WAIT_TOKEN).strip()
    else:
        # cprint(string.replace(WAIT_TOKEN, '').strip(), color='green')
        return string.replace(WAIT_TOKEN, '').strip()


def add_period(s):
    if s and s[-1] not in '.,!?':
        s += '.'
    return s


def remove_ALL_wait_tokens(input_string):
    # remove "__ " from the beginning of the string
    output_string = input_string.replace('▁▁', '').strip()
    output_string = output_string.replace('  ', '').strip()
    return output_string


def remove_all_but_last_wait_token(input_string):
    # remove "__ " from the beginning of the string
    output_string = re.sub(f'^{WAIT_TOKEN} ', '', input_string)

    # remove " __ " (and its multiple occurrences) from the middle of the string
    output_string = re.sub(f'({WAIT_TOKEN} )+', '', output_string)

    # remove remaining "space+WAIT_TOKEN+comma"
    output_string = output_string.replace(" ▁▁,", ",")

    return output_string


def _torch_collate_batch(
    examples,
    tokenizer,
    pad_to_multiple_of: Optional[int] = None,
    sample: Optional[bool] = False,
):
    """
    Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary.
    THE ONLY DIFF FROM THE ORIG FUNC IS SAMPLING PREFIXES
    """

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError("You are attempting to pad samples but the tokenizer you are using"
                         f" ({tokenizer.__class__.__name__}) does not have a pad token.")

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, :example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


class DataCollatorMixin:

    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)
        # if return_tensors is None:
        #     return_tensors = self.return_tensors
        # elif return_tensors == "pt":
        #     return self.torch_call(features)
        # else:
        #     raise NotImplementedError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class SamplingDataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    logweights: bool = False
    detokenizer = TreebankWordDetokenizer()
    debug: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead.")

    def preprocess_and_sample(self, data_point):
        """ Randomly sample the beginning of a sentence (prefix), and tokenize """
        src_sent = data_point['src'].strip()
        tgt_sent = data_point['tgt'].strip()

        src_words = nltk.word_tokenize(src_sent)
        tgt_words = nltk.word_tokenize(tgt_sent)

        full_length = len(src_words)

        # NOTE: maybe sample longer sequences more.
        if self.logweights:
            try:
                logspace = np.logspace(0, 1, full_length)
                p = logspace / logspace.sum()
                sample_length = np.random.choice(range(full_length), p=p)
                sample_length = int(np.clip(sample_length, a_min=1, a_max=sample_length))  # NOTE: investigate this
            except Exception as e:
                sample_length = np.random.randint(full_length)
                print(e)
        else:
            sample_length = np.random.randint(full_length)

        src_sent = self.detokenizer.detokenize(src_words[:sample_length + 1])
        tgt_sent = self.detokenizer.detokenize(tgt_words[:sample_length + 1])

        src_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', src_sent)
        tgt_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', tgt_sent)

        src_sent = src_sent.replace(f"{WAIT_TOKEN}", "").strip()  # NOTE: remove all the wait tokens in the source
        tgt_sent = remove_all_but_last_wait_token(tgt_sent)  # remove all but the last wait token (if it is last)

        if self.debug:
            c = 'red' if (sample_length == full_length - 1) else "grey"
            cprint(src_sent, color=c, attrs=['bold'])
            cprint(tgt_sent, color=c)

        # only add EOS if the tranlation if finished
        if sample_length == full_length - 1:
            src_sent = add_period(src_sent)  # add period if no period, exclamation or question mark
            string = (f"{self.tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent} {self.tokenizer.eos_token}")
        else:
            string = (f"{self.tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent}")
        src = self.tokenizer(string, add_special_tokens=False)

        return {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}

    def torch_call(self, examples: List[Dict[str, str]]) -> Dict[str, str]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            examples = list(map(self.preprocess_and_sample, examples))
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            raise NotImplementedError('Must be a dict')
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            raise NotImplementedError("mlm NOT implemented for this custom collator.")
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"],
                                                                         special_tokens_mask=special_tokens_mask)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        return inputs, labels


@dataclass
class SamplingDataCollatorForLanguageModelingJA(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    WAIT_TOKEN: str
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    logweights: bool = False
    debug: bool = False
    _DEBUG_COUNTER = 0
    japanese_target: bool = False
    mask_prompt: bool = False

    def __post_init__(self):
        self.tokenize_source = EuTokenizer().tokenize
        self.tokenize_target = JaTokenizer().tokenize_whole_sentence if self.japanese_target else self.tokenize_source
        self.detokenize_source = EuTokenizer().detokenize
        self.detokenize_target = JaTokenizer().detokenize if self.japanese_target else self.detokenize_source
        cprint(f"USING JAPANESE TARGETS: {self.japanese_target}", 'yellow', 'on_green')
        cprint(f"WAIT_TOKEN: {self.WAIT_TOKEN}", 'yellow', 'on_green')
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead.")

    def preprocess_and_sample(self, data_point):
        """ Randomly sample the beginning of a sentence (prefix), and tokenize """
        src_sent = data_point['src'].strip()
        tgt_sent = data_point['tgt'].strip()

        src_words = self.tokenize_source(src_sent)
        tgt_words = self.tokenize_target(tgt_sent)

        full_length = get_num_words_wo_fillers(src_words, self.WAIT_TOKEN)

        if self.debug:
            cprint(f'words minus fillers: {full_length}, full_length: {len(src_words)}', 'yellow', 'on_blue')
        full_length_with_fillers = len(src_words)

        # NOTE: maybe sample longer sequences more.
        if self.logweights:
            try:
                logspace = np.logspace(0, 1, full_length)
                p = logspace / logspace.sum()
                sample_length = np.random.choice(range(full_length), p=p)
                sample_length = int(np.clip(sample_length, a_min=1, a_max=sample_length))  # NOTE: investigate this
            except Exception as e:
                sample_length = np.random.randint(full_length)
                print(e)
        else:
            sample_length = np.random.randint(full_length)

        src_sent = self.detokenize_source(src_words[:sample_length + 1])
        src_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', src_sent)
        src_sent = dropAllWaitTokens(
            fix_spaces_around_punctuation(src_sent),
            self.WAIT_TOKEN,
        )  # NOTE: remove wait tokens in the source
        if self.debug:
            cprint(f'sample_length: {sample_length}', 'green', 'on_yellow')
            # cprint(src_sent, color='magenta')
        if not src_sent.endswith('.'):
            tgt_sent = fix_spaces_around_punctuation(
                remove_multiple_spaces(
                    leave_only_last_wait_token(
                        self.detokenize_target(tgt_words[:sample_length + 1]),
                        self.WAIT_TOKEN,
                    )))
        else:
            if self.debug:
                cprint('DROPPING ALL WAIT TOKENS, RETURNING THE FULL TARGET', color='cyan')
            tgt_sent = fix_spaces_around_punctuation(
                remove_multiple_spaces(dropAllWaitTokens(
                    self.detokenize_target(tgt_words),
                    self.WAIT_TOKEN,
                )))

        # FIXME: this ugly hack (removes unnecessary spaces in Japanese)
        if self.japanese_target:
            if tgt_sent.endswith(self.WAIT_TOKEN):
                tgt_sent = tgt_sent.replace(" ", "").replace(self.WAIT_TOKEN, f" {self.WAIT_TOKEN}")
            else:
                tgt_sent = tgt_sent.replace(" ", "")
            tgt_sent = tgt_sent.replace("、、", "、")

        # tgt_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', tgt_sent)

        if self.debug:
            c = 'red' if (sample_length == full_length - 1) else "grey"
            # print(src_words[:sample_length + 1])
            # print(tgt_words[:sample_length + 1])
            cprint(f"ID: {self._DEBUG_COUNTER}: {src_sent}", color=c, attrs=['bold'])
            cprint(f"ID: {self._DEBUG_COUNTER}: {tgt_sent}", color=c)
            self._DEBUG_COUNTER += 1

        # only add EOS if the tranlation if finished
        if sample_length == full_length - 1:
            src_sent = add_period(src_sent)  # add period if no period, exclamation or question mark
            string = (f"{self.tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent} {self.tokenizer.eos_token}")
        else:
            string = (f"{self.tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                      f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent}")
        src = self.tokenizer(string, add_special_tokens=False)

        return {'input_ids': src['input_ids'], 'attention_mask': src['attention_mask']}

    def torch_call(self, examples: List[Dict[str, str]]) -> Dict[str, str]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            examples = list(map(self.preprocess_and_sample, examples))
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            raise NotImplementedError('Must be a dict')
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            raise NotImplementedError("mlm NOT implemented for this custom collator.")
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"],
                                                                         special_tokens_mask=special_tokens_mask)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            if self.mask_prompt:
                labels[:, :103] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        return inputs, labels


def PromptUpdater(src_sent, _tokenizer):

    def src_prefix_generator(src_sent):
        src_words = nltk.word_tokenize(src_sent)
        # src_words = nltk.word_tokenize(src_sent[:-4])  # it's a workaround: NLTK doesn't recognize </s> as a word
        # src_words.append(src_sent[-4:])  # it's a workaround: NLTK doesn't recognize </s> as a word
        i = 0
        while True:
            i += 1
            yield src_words[:i]

    def update_prompt_with_new_source_and_target(tgt_sent):
        nonlocal src_prefix_generator, tok  # make the generator and tokenizer visible from outer scope
        src_prefix = next(src_generator)
        src_sent = re.sub(r'\s([,.!?;](?:\s|$))', r'\1', ' '.join(src_prefix))

        string = (f"{tok.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
                  f"Translate this text: \"{src_sent}\" {E_INST} {tgt_sent}")

        tokenized = tok(string, add_special_tokens=False, return_tensors='pt')

        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}

    src_generator = src_prefix_generator(
        src_sent)  # init generator that will persist across calls to update_prompt_with_new_source_and_target
    tok = _tokenizer
    return update_prompt_with_new_source_and_target  # return the function
