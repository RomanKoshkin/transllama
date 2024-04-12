import os, sys, time
from dotenv import load_dotenv

load_dotenv('../.env', override=True)  # load API keys into
print(os.getenv("HF_HOME"))
time.sleep(3)
sys.path.append('../')
import importlib, argparse, re, torch
import openai, time, tiktoken
from collections import deque
from itertools import zip_longest
from pprint import pformat
from termcolor import cprint
from utils.llm_utils import fix_spaces_around_punctuation
from utils.utils import Timer

# from configs import config_1 as CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=int, default=-1)
parser.add_argument("--asr_model", type=str, default="whisper-large-v2")
parser.add_argument("--min_lag", type=int, default=-1)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--wait_k", type=int, default=None)
args, unknown_args = parser.parse_known_args()

ASR_MODEL_NAME = f"openai/{args.asr_model}"  # "openai/whisper-small.en"
DEVICE = torch.cuda.current_device()
SRATE = 16000  # FIXME
MIN_LAG_WORDS = args.min_lag

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id) # NOTE: IMPORTANT: set in the entry script se.py
CONFIG_ID = args.config_id
spec = importlib.util.spec_from_file_location("config", f'../configs/config_{CONFIG_ID}.py')
CONFIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG)

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent, SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction

import torch

from termcolor import cprint
from utils.datasets import JaTokenizer
from utils.llm_utils import join_punctuation_ja

from transformers import WhisperProcessor, WhisperForConditionalGeneration

ja_tokenizer = JaTokenizer()

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

# openai.api_key = os.getenv("OPENAI_KEY_AHC")
openai.api_key = os.getenv("OPENAI_KEY_MINE")

ONLINE_TRANSLATION_PATH = f"ONLINE_TARGETS/{MODEL_NAME}_{TARGET_LANGUAGE}_online_target_config_{args.config_id}.txt"
if os.path.exists(ONLINE_TRANSLATION_PATH):
    os.remove(ONLINE_TRANSLATION_PATH)
WAIT_PATH = f"WAIT_TOKENS/wait_config_{args.config_id}.txt"
if os.path.exists(WAIT_PATH):
    os.remove(WAIT_PATH)
ASR_PATH = f"ASR_RESULTS/asr_config_{CONFIG_ID}.txt"
if os.path.exists(ASR_PATH):
    os.remove(ASR_PATH)

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

MODEL_NAME = CONFIG.MODEL_NAME
TARGET_LANGUAGE = CONFIG.TARGET_LANGUAGE

# for backward compatibility with old configs
try:
    japanese_target = CONFIG.japanese_target
except:
    cprint("Setting `japanese_target` to False", 'red', 'on_green')
    japanese_target = False
cprint(f"japanese_target: {japanese_target}", 'red', 'on_green')
time.sleep(5)

cprint(f"LANG: {TARGET_LANGUAGE} | japanese_target: {japanese_target}", 'grey', 'on_red', attrs=['bold'])
SPACE = "" if TARGET_LANGUAGE == 'Japanese' else " "
PERIOD = "。" if TARGET_LANGUAGE == 'Japanese' else "."
# --------------------------------------------------------------------------------------------------------------------


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
    global ABORT_SENTENCE

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
    if bool(re.search(r'[\'~@#\$%&\+=\"\]\[\(\)\|▁_\*\^<>\{\}\\\/]', tgt)) or contains_latin_chars(tgt):
        cprint('WAIT_TOKEN DETECTED', 'magenta', 'on_green', attrs=['bold'])
        with open(WAIT_PATH, 'a') as f:
            f.write(tgt + '\n')
        tgt = ""

    cprint(tgt, 'black', 'on_red')
    punct_after_end_of_source = False
    if not source_finished:
        tgt = tgt.strip().replace('.', '').replace('?', '').strip()
    else:
        for i in ['.', '?', '!']:
            if i in tgt[-30:]:
                tgt = tgt.split(i)[0] + i
                punct_after_end_of_source = True
                break

    return tgt, punct_after_end_of_source


def find_repeating_patterns(s, min_length=3):
    """
    Find repeating patterns in a string and count their occurrences.
    
    :param s: The input string.
    :param min_length: Minimum length of repeating pattern to consider.
    :return: A dictionary with patterns as keys and their counts as values.
    """
    pattern_counts = {}
    for length in range(min_length, len(s) // 2 + 1):
        for match in re.finditer(r'(?=(.{%d,}?)\1)' % length, s):
            pattern = match.group(1)
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1
    return pattern_counts


def has_repeated_chars(s, count=3):
    return bool(re.search(r'(.)\1{%d,}' % (count - 1), s))


def check_for_loop(s, max_repeated_chars=3):
    patterns = find_repeating_patterns(s)
    if has_repeated_chars(s, count=max_repeated_chars):
        return True
    if len(patterns) == 0:
        return False
    for p, c in patterns.items():
        if (len(p) >= 1) and (c > 5):
            return True
        else:
            return False


def earliest_occurrence(searched_list, search_items):
    for idx, value in enumerate(searched_list):
        if value in search_items:
            return idx
    return None  # Return None if no occurrence is found


def get_new_content(old_string, new_string):

    len_old = len(old_string)
    len_new = len(new_string)
    try:
        assert len_old <= len_new
    except Exception as e:
        cprint(f'old_string: {old_string}', color='yellow')
        cprint(f'new_string: {new_string}', color='yellow')
        raise e
    return new_string[len_old:]


def new_items(old_list, new_list):
    return [b for a, b in zip_longest(old_list, new_list) if not a]


def decide_if_finished(tgt, ABORT_SENTENCE, punct_after_end_of_source):
    if PERIOD in tgt[-30:]:
        return True
    if ABORT_SENTENCE:
        return True
    if punct_after_end_of_source:
        return True
    return False


class ASR:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
        self.asr_model.config.forced_decoder_ids = None
        self.srate = SRATE
        self.DEVICE = DEVICE

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(' ') if not (i.endswith('-') or i.endswith('...'))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith('.'):
                s[-1] += '.'
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(','):
                        if not s[i + 1].startswith('I'):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array, source_finished):
        input_features = self.processor(
            audio_array,
            sampling_rate=self.srate,
            return_tensors="pt",
        ).input_features.to(self.DEVICE)
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return self._postprocess(transcription, source_finished)


@entrypoint
class Agent(SpeechToTextAgent):

    TRANSLATION_TOKENS = []
    asr_model = ASR(ASR_MODEL_NAME, DEVICE, SRATE)
    input_words = []
    TGT = ""
    _japanese_target = False
    WAIT_TOKENS_IN_SENTENCE = 0

    def policy(self):
        global ABORT_SENTENCE, SPACE, PERIOD

        try:
            if len(self.TRANSLATION_TOKENS) == 0:
                self.start_time = time.time()

            finished = False

            input_words = self.asr_model.recognize(
                self.states.source,
                self.states.source_finished,
            )
            # get the new words added on top of the previously recognized ones
            new_words = new_items(self.input_words, input_words)  # get new words from the full list of sentence words
            if not self.states.source_finished:
                new_words = [w.strip().replace('.', '').replace('?', '').replace('!', '') for w in new_words]

            cprint(f"Old words: {self.input_words}", 'blue', 'on_white', attrs=['bold'])
            cprint(f"ASR output: {input_words}", 'grey', 'on_white', attrs=['bold'])
            cprint(f"New words: {new_words}", 'green', 'on_white', attrs=['bold'])

            if len(new_words) > 0:
                self.input_words.append(new_words[0])  # append only the first word
            cprint(f"Updated words: {self.input_words}", 'red', 'on_white', attrs=['bold'])

            # if the ASR returned no new words, read in more
            if (len(new_words) == 0) or ((len(self.input_words) < WAIT_K) and not self.states.source_finished):
                if not self.states.source_finished:
                    cprint('resetting WAIT_TOKENS_IN_SENTENCE, returning ReadAction()', color='magenta', attrs=['bold'])
                    self.WAIT_TOKENS_IN_SENTENCE = 0
                    return ReadAction()

            src_text = " ".join(self.input_words)  # join source words into text (words are with punctuation)
            tgt, punct_after_end_of_source = call_api(src_text, self.TGT, self.states.source_finished)
            self.TGT = fix_spaces_around_punctuation(SPACE.join([self.TGT, tgt]).strip())
            finished = decide_if_finished(self.TGT, ABORT_SENTENCE, punct_after_end_of_source)
            wait = True if tgt == "" else False
            if ABORT_SENTENCE:
                wait = False
            if wait:
                self.WAIT_TOKENS_IN_SENTENCE += 1
            cprint(f'{self.TGT.endswith(PERIOD)}, {self.TGT}, finished: {finished}, wait: {wait}', color='yellow')

            # stop if all the source words have been revealed, but target is 2x the source
            if self.states.source_finished:
                ratio = len(self.TGT.split(' ')) / (0.1 + len(self.input_words))
                cprint(f"wait: {wait} | finished: {finished} | ratio: {ratio}", color='green')
                if ratio > 3:
                    finished = True
                    wait = False
                    print()
                    cprint('Forcefully terminated the sentence', color='yellow')
                    time.sleep(5)
                    print()

            if finished:
                self.TRANSLATION_TOKENS = []  # clear the list of translation tokens
                with open(ONLINE_TRANSLATION_PATH, 'a', encoding='utf-8') as f:
                    f.write(self.TGT + "\n")
                with open(ASR_PATH, 'a', encoding='utf-8') as f:
                    f.write(" ".join(self.input_words) + "\n")
                self.input_words = []

            cprint(f'{src_text} : {self.TGT}', color='blue')
            cprint(
                f'WAIT: {wait} FINISHED: {finished} SRC_FINISHED: {self.states.source_finished} len_states_source:  {len(self.states.source)}',
                color='magenta')

            if not wait:
                last_word = self.TGT.strip().split(' ')[-1]
                self.TGT = "" if finished else self.TGT
                return WriteAction(last_word, finished=finished)
            elif wait and not ABORT_SENTENCE:
                return ReadAction()
            else:
                cprint('ERROR', color='yellow')
                print(wait, ABORT_SENTENCE, finished)
                time.sleep(int(1e10))

        except Exception as e:
            cprint(f'ERROR: {e}', color='yellow')
            self.TRANSLATION_TOKENS = []  # class property
            self.WAIT_TOKENS_IN_SENTENCE = 0  # global var
            return WriteAction(f'', finished=True)