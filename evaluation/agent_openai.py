import argparse, importlib
from argparse import Namespace
from collections import deque
import os, sys, time, re
from typing import Optional
import openai
import time
import tiktoken
from pprint import pformat
from termcolor import cprint
from dotenv import load_dotenv
from utils.utils import Timer

load_dotenv('../.env', override=True)  # load API keys into
print(os.getenv("HF_HOME"))
sys.path.append('../')

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from termcolor import cprint
from utils.llm_utils import fix_spaces_around_punctuation

WAIT_TOKENS_IN_SENTENCE = 0

parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=int, default=None)
args, unknown_args = parser.parse_known_args()

spec = importlib.util.spec_from_file_location("config", f'../configs/config_{args.config_id}.py')
CONFIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG)

parser.add_argument("--wait_k", type=int, default=CONFIG.WAIT_K)
args, unknown_args = parser.parse_known_args()

MODEL_NAME = CONFIG.MODEL_NAME
TARGET_LANGUAGE = CONFIG.TARGET_LANGUAGE
MAX_TOKENS = CONFIG.MAX_TOKENS
TEMPERATURE = CONFIG.TEMPERATURE
WAIT_TOKEN = CONFIG.WAIT_TOKEN
WAIT_K = args.wait_k
SYS_PROMPT_INIT = CONFIG.SYS_PROMPT_INIT
SYS_PROMPT = CONFIG.SYS_PROMPT
SPACE = "" if TARGET_LANGUAGE == 'Japanese' else " "
PERIOD = "。" if TARGET_LANGUAGE == 'Japanese' else "."

openai.api_key = os.getenv("OPENAI_KEY_AHC")
# openai.api_key = os.getenv("OPENAI_KEY_MINE")
# openai.api_key = os.getenv("OPENAI_KEY_ANDREI")

ONLINE_TRANSLATION_PATH = f"ONLINE_TARGETS/{MODEL_NAME}_{TARGET_LANGUAGE}_online_target_config_{args.config_id}.txt"
if os.path.exists(ONLINE_TRANSLATION_PATH):
    os.remove(ONLINE_TRANSLATION_PATH)
WAIT_PATH = f"WAIT_TOKENS/wait_config_{args.config_id}.txt"
if os.path.exists(WAIT_PATH):
    os.remove(WAIT_PATH)

enc = tiktoken.encoding_for_model(MODEL_NAME)

REV_VOCAB = {i: enc.decode_single_token_bytes(i) for i in range(100256)}
A, B = [], []
for ID, tok in REV_VOCAB.items():
    try:
        bpe = tok.decode()
        if bpe.endswith('。' if TARGET_LANGUAGE == "Japanese" else '.'):
            B.append(ID)
        else:
            A.append(ID)
    except:
        A.append(ID)

LOGIT_BIASES_NOT_FINISHED = {b: -100 for b in B}

A, B = [], []
for ID, tok in REV_VOCAB.items():
    try:
        bpe = tok.decode()
        if WAIT_TOKEN in bpe:
            B.append(ID)
        else:
            A.append(ID)
    except:
        A.append(ID)

LOGIT_BIASES_FINISHED = {b: -100 for b in B}
LOGIT_BIASES_FINISHED[1811] = 0.2  # NOTE: promote full stop when source is finished
LOGIT_BIASES_FINISHED[26] = -50  # NOTE: demote semi-colons when source is finished
LOGIT_BIASES_FINISHED[92653] = -50  # muliple periods
LOGIT_BIASES_FINISHED[1981] = -50  # triple dots

ABORT_SENTENCE = False
PREVIOUS_CONTENT = deque([1, 1, 1, 1], maxlen=4)
cprint(f"WAIT_K: {WAIT_K}", 'black', 'on_red', attrs=['bold'])


def return_first_word(tgt):
    words = tgt.strip().split(SPACE)
    return words[0]


def trim_off_incomplete(tgt):
    words = tgt.strip().split(SPACE)
    if len(words) > 1:
        return SPACE.join(words[:-1])
    else:
        return words[0]


contains_latin_chars = lambda v: bool(re.search('[a-zA-Z]', v)) if TARGET_LANGUAGE in ['Russian', 'Japanese'] else False


def call_api(src, TGT, source_finished):
    global PREVIOUS_CONTENT, ABORT_SENTENCE

    prompt = f'Source: {src}\nTarget: {TGT}'
    messages = [{
        "role": "system",
        "content": SYS_PROMPT if len(src) > WAIT_K else SYS_PROMPT_INIT,
    }, {
        "role": "user",
        "content": prompt,
    }]
    print(messages)

    if len(src) > WAIT_K:
        max_tokens = MAX_TOKENS
    elif len(src) == WAIT_K:
        max_tokens = 30
    # elif len(src) == len(SRC.split(" ")):
    #     max_tokens = 100
    else:
        pass

    while True:
        try:
            time.sleep(0.1)
            print('calling api...')
            with Timer():
                r = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    presence_penalty=1.0,  # positives penalize repetitiveness [-2; 2]
                    max_tokens=max_tokens,
                    stop=["Source:", "Target:", " "],
                    logit_bias=LOGIT_BIASES_NOT_FINISHED if not source_finished else LOGIT_BIASES_FINISHED,
                )
            break
        except Exception as e:
            cprint(e, 'grey', 'on_yellow')
            cprint('Rate limit exceeded. Sleeping for one minute.', 'yellow', "on_red")
            time.sleep(60)

    content = r.choices[0].message.content
    PREVIOUS_CONTENT.append(content)
    cprint(PREVIOUS_CONTENT, color='green')
    cprint(content, color='red', attrs=['bold'])
    ABORT_SENTENCE = True if all([content == pc for pc in PREVIOUS_CONTENT]) else False
    cprint(f'ABORT_SENTENCE: {ABORT_SENTENCE}', color='blue', attrs=['bold'])

    if TARGET_LANGUAGE != 'Japanese':
        if len(src) == WAIT_K:
            tgt = trim_off_incomplete(content)
        else:
            tgt = return_first_word(content)
    else:
        tgt = content

    # remove wait tokens NOTE (!!!) we assume that the WAIT_TOKEN contains either @ or ▁
    if bool(re.search(r'[\'~@#\$%&\+=\"\]\[\(\)\|▁_\*\^<>\{\}\?!\\\/]', tgt)) or contains_latin_chars(tgt):
        cprint('WAIT_TOKEN DETECTED', 'magenta', 'on_green', attrs=['bold'])
        with open(WAIT_PATH, 'a') as f:
            f.write(f"__content__: {content} __tgt__: {tgt}\n")
        tgt = ""

    return tgt


@entrypoint
class Agent(TextToTextAgent):

    def __init__(self, args: Optional[Namespace] = None) -> None:
        super().__init__(args)
        self.TGT = ""
        self.WAIT_TOKENS_IN_SENTENCE = 0
        self._japanese_target = False

    def policy(self):
        global ABORT_SENTENCE, SPACE, PERIOD

        src = " ".join(self.states.source)

        if len(self.states.source) == 0:
            cprint('resetting WAIT_TOKENS_IN_SENTENCE', color='magenta', attrs=['bold'])
            self.WAIT_TOKENS_IN_SENTENCE = 0

        if (len(self.states.source) < WAIT_K) and not self.states.source_finished:
            return ReadAction()
        else:
            tgt = call_api(src, self.TGT, self.states.source_finished)
            self.TGT = fix_spaces_around_punctuation(SPACE.join([self.TGT, tgt]).strip())

            finished = True if (PERIOD in self.TGT[-30:]) or ABORT_SENTENCE else False  # FIXME: this might break
            wait = True if tgt == "" else False
            if wait:
                self.WAIT_TOKENS_IN_SENTENCE += 1
            cprint(f'{self.TGT.endswith(PERIOD)}, {self.TGT}, finished: {finished}, wait: {wait}', color='yellow')

            # if wait token, read in more
            if wait and not ABORT_SENTENCE:
                return ReadAction()

            if len(self.TGT) > (len(src) * 4):
                cprint("length of target is already 4x that of the source. Aborting.", 'grey', 'on_yellow')
                self.TGT += PERIOD
                finished = True

            if finished:
                self.WAIT_TOKENS_IN_SENTENCE = WAIT_TOKENS_IN_SENTENCE
                cprint(self.WAIT_TOKENS_IN_SENTENCE, 'grey', 'on_cyan', attrs=['bold'])
                with open(ONLINE_TRANSLATION_PATH, 'a', encoding='utf-8') as f:
                    f.write(self.TGT + "\n")
                self.TGT = ""

            # either way:
            cprint(
                f'FINISHED: {finished} SRC_FINISHED: {self.states.source_finished} len_states_source:  {len(self.states.source)}',
                color='magenta')

            return WriteAction(tgt, finished=finished)