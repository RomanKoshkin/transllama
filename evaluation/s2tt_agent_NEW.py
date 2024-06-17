import os, sys, importlib, time, argparse, re
from itertools import zip_longest
from pprint import pformat
from termcolor import cprint
from dotenv import load_dotenv

load_dotenv("../.env", override=True)  # load API keys into
print(os.getenv("HF_HOME"))
sys.path.append("../")

# from configs import config_1 as CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=int, default=-1)
parser.add_argument("--asr_model", type=str, default="")
parser.add_argument("--min_lag", type=int, default=-1)
parser.add_argument("--min_read_time", type=float, default=0)
args, unknown_args = parser.parse_known_args()

MIN_READ_TIME = args.min_read_time
ASR_MODEL_NAME = f"openai/{args.asr_model}"  # "openai/whisper-small.en"
SRATE = 16000  # FIXME
MIN_LAG_WORDS = args.min_lag

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id) # NOTE: IMPORTANT: set in the entry script se.py
CONFIG_ID = args.config_id
spec = importlib.util.spec_from_file_location(
    "config", f"../configs/config_{CONFIG_ID}.py"
)
CONFIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIG)

from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent, SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction

import torch, copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training

from termcolor import cprint
from utils.utils import AttributeDict
from utils.datasets import JaTokenizer
from utils.llm_utils import join_punctuation_ja

from transformers import WhisperProcessor, WhisperForConditionalGeneration

ja_tokenizer = JaTokenizer()

B_INST = CONFIG.B_INST
E_INST = CONFIG.E_INST
B_SYS = CONFIG.B_SYS
E_SYS = CONFIG.E_SYS
DEFAULT_SYSTEM_PROMPT = CONFIG.DEFAULT_SYSTEM_PROMPT
# --------------------------------------------------------------------------------------------------------------------
GENERATION_CONFIG = CONFIG.GENERATION_CONFIG
BNB_CONFIG = CONFIG.BNB_CONFIG
DEVICE = torch.cuda.current_device()
MODEL_NAME = CONFIG.MODEL_NAME
ADAPTER_NAME = CONFIG.ADAPTER_NAME
PREDICT_TOKEN_BY_TOKEN = CONFIG.PREDICT_TOKEN_BY_TOKEN
WAIT_TOKEN_IDS = CONFIG.WAIT_TOKEN_IDS
MAX_WAIT_TOKEN_COUNT = CONFIG.MAX_WAIT_TOKEN_COUNT
SUPPRESSED_TOKENS_UNFINISHED_INPUT = CONFIG.SUPPRESSED_TOKENS_UNFINISHED_INPUT
SUPPRESSED_TOKENS_FINISHED_INPUT = CONFIG.SUPPRESSED_TOKENS_FINISHED_INPUT
TARGET_LANGUAGE = CONFIG.TARGET_LANGUAGE

# for backward compatibility with old configs
try:
    japanese_target = CONFIG.japanese_target
except:
    cprint("Setting `japanese_target` to False", "red", "on_green")
    japanese_target = False
cprint(f"japanese_target: {japanese_target}", "red", "on_green")
time.sleep(5)
tokenizer = CONFIG.tokenizer
REV_VOCAB = CONFIG.REV_VOCAB

# fianlly, parse additional parameters to override generation configs
parser.add_argument("--num_beams", type=int, default=CONFIG.NUM_BEAMS)
parser.add_argument("--beam_depth", type=int, default=CONFIG.BEAM_DEPTH)
parser.add_argument("--wait_k", type=int, default=CONFIG.WAIT_K)
parser.add_argument(
    "--zero_shot",
    action="store_true",
    default=False,
    help="Use base model. Default: false",
)
args, unknown_args = parser.parse_known_args()

NUM_BEAMS = args.num_beams
BEAM_DEPTH = args.beam_depth
cprint(f"NUM_BEAMS: {NUM_BEAMS}", "grey", "on_blue")
cprint(f"BEAM_DEPTH: {BEAM_DEPTH}", "grey", "on_blue")
cprint(f"do_sample: {GENERATION_CONFIG.do_sample}", "grey", "on_blue")
WAIT_K = args.wait_k
if not args.zero_shot:
    ONLINE_TRANSLATION_PATH = f"ONLINE_TARGETS/online_target_{CONFIG_ID}.txt"
else:
    ONLINE_TRANSLATION_PATH = (
        f"ONLINE_TARGETS/online_target_config_{CONFIG_ID}_zero_shot.txt"
    )
if os.path.exists(ONLINE_TRANSLATION_PATH):
    os.remove(ONLINE_TRANSLATION_PATH)
ASR_PATH = f"ASR_RESULTS/asr_config_{CONFIG_ID}.txt"
if os.path.exists(ASR_PATH):
    os.remove(ASR_PATH)
WAIT_PATH = f"WAIT_TOKENS/wait_config_{CONFIG_ID}.txt"
if os.path.exists(WAIT_PATH):
    os.remove(WAIT_PATH)

cprint(
    f"LANG: {TARGET_LANGUAGE} | japanese_target: {japanese_target}",
    "grey",
    "on_red",
    attrs=["bold"],
)
# --------------------------------------------------------------------------------------------------------------------


class DummyModel:

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Identity()
        self.counter = 0
        self.GENERATION_CONFIG = AttributeDict()

    def forward(self, x):
        return self.layer(x)

    def generate(self, **kwargs):
        if self.counter == 16:
            self.counter = 0
            return torch.LongTensor([2]).view(1, 1)
        else:
            self.counter += 1
            return torch.randint(low=3, high=30000, size=(1, 1))


# model = DummyModel()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,  # "auto" to put different layers of the model to diff devs as necessary
    trust_remote_code=True,
    quantization_config=None if DEVICE == "cpu" else BNB_CONFIG,
)

# comment test the base model (before fine-tuning)
if not args.zero_shot:
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, model_id=ADAPTER_NAME)


def find_repeating_patterns(s, min_length=3):
    """
    Find repeating patterns in a string and count their occurrences.

    :param s: The input string.
    :param min_length: Minimum length of repeating pattern to consider.
    :return: A dictionary with patterns as keys and their counts as values.
    """
    pattern_counts = {}
    for length in range(min_length, len(s) // 2 + 1):
        for match in re.finditer(r"(?=(.{%d,}?)\1)" % length, s):
            pattern = match.group(1)
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1
    return pattern_counts


def has_repeated_chars(s, count=3):
    return bool(re.search(r"(.)\1{%d,}" % (count - 1), s))


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


def make_prompt(src_text, TRANSLATION_TOKENS):
    """returns a prompts given some source text and TRANSLATION_TOKENS (generated so far)"""
    tgt_text = tokenizer.decode(
        TRANSLATION_TOKENS, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )
    prompt = (
        f"{tokenizer.bos_token}{B_INST}{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}"
        f'Translate this text: "{src_text}" {E_INST} {tgt_text}'
    )
    return prompt


def get_complete_words_from_token_list_ja(trg_tokens):
    """returns complete words given a list of llama tokens"""
    global ja_tokenizer, tokenizer, WAIT_TOKEN_ID
    string = tokenizer.decode(
        trg_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    return join_punctuation_ja(ja_tokenizer.get_full_words(string))


def get_complete_words_from_token_list(trg_tokens):
    """returns complete words given a list of tokens"""
    trg_tokens = [1483 if i == tokenizer.eos_token_id else i for i in trg_tokens]  # ▁Р

    word_parts = [REV_VOCAB[t] for t in trg_tokens]
    indices = list(
        filter(
            None, [i if p.startswith("▁") else None for i, p in enumerate(word_parts)]
        )
    )
    if indices:
        chunks = [
            "".join(word_parts[i:j]).lstrip("▁")
            for i, j in zip([0] + indices, indices + [None])
        ]
        if chunks[-1].endswith("."):
            return list(filter(None, chunks))  # filter empty strings
        else:
            return list(filter(None, chunks[:-1]))  # filter empty strings
    else:
        return list()


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
        cprint(f"old_string: {old_string}", color="yellow")
        cprint(f"new_string: {new_string}", color="yellow")
        raise e
    return new_string[len_old:]


def predict_next_word(src_text, TRANSLATION_TOKENS, source_finished=None):
    """Predict as many tokens as needed until a full word is formed"""
    global PREVIOUS_FULL_WORDS, WAIT_TOKEN_COUNT, GENERATION_CONFIG, WAIT_TOKENS_IN_SENTENCE

    _begining_new_word = True

    while True:

        prompt = make_prompt(src_text, TRANSLATION_TOKENS)
        encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
            DEVICE
        )
        cprint(prompt, color="red")
        with open("prompt.log", "a") as f:
            f.write(prompt + "\n")

        if not source_finished:
            GENERATION_CONFIG.suppress_tokens = SUPPRESSED_TOKENS_UNFINISHED_INPUT
            GENERATION_CONFIG.num_beams = NUM_BEAMS if _begining_new_word else 1
            GENERATION_CONFIG.max_new_tokens = BEAM_DEPTH if _begining_new_word else 1
            print("suppressing period and EOS")
        else:
            cprint("source_finished", "yellow", "on_green")
            GENERATION_CONFIG.suppress_tokens = SUPPRESSED_TOKENS_FINISHED_INPUT
            GENERATION_CONFIG.num_beams = NUM_BEAMS if _begining_new_word else 1
            GENERATION_CONFIG.max_new_tokens = BEAM_DEPTH if _begining_new_word else 1

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                generation_config=GENERATION_CONFIG,
            )
        _begining_new_word = False  # self this flag to say we're mid-word
        last_token = outputs[0][-GENERATION_CONFIG.max_new_tokens].item()

        complete_words_predicted = get_complete_words_from_token_list(
            TRANSLATION_TOKENS
        )

        if (
            source_finished
        ):  # finished if either EOS or . (w/o a leading space) is generated
            if (last_token == tokenizer.eos_token_id) or (
                complete_words_predicted[-1].strip().endswith(".")
            ):
                finished = True
                wait = False
                PREVIOUS_FULL_WORDS = []
                cprint("FINISHED", "red", "on_grey")
                break  # return the full translation

        if last_token not in WAIT_TOKEN_IDS:
            WAIT_TOKEN_COUNT = 0
            TRANSLATION_TOKENS.append(last_token)
        else:
            cprint("WAIT TOKEN generated", "red", "on_yellow")
            with open(WAIT_PATH, "a") as f:
                f.write(prompt + "\n")
            WAIT_TOKENS_IN_SENTENCE += 1
            WAIT_TOKEN_COUNT += 1
            if WAIT_TOKEN_COUNT > MAX_WAIT_TOKEN_COUNT:
                finished = True
                wait = False
                WAIT_TOKEN_COUNT = 0
            else:
                finished = False
                wait = True
            break  # return for more source words (READ)

        if (len(TRANSLATION_TOKENS) > 0) and (TRANSLATION_TOKENS[-1] in [2]):
            finished = True
            wait = False
            PREVIOUS_FULL_WORDS = []
            cprint("FINISHED", "red", "on_grey")
            break  # return the full translation

        if len(complete_words_predicted) > len(PREVIOUS_FULL_WORDS):
            print(complete_words_predicted, TRANSLATION_TOKENS)
            cprint(
                f"EMITTING NEW FULL WORD: {complete_words_predicted}",
                "grey",
                "on_yellow",
            )
            PREVIOUS_FULL_WORDS = copy.deepcopy(complete_words_predicted)
            TRANSLATION_TOKENS.pop(
                -1
            )  # remove the last incomplete word to restart with the complete one
            finished = False
            wait = False
            break

        if check_for_loop(
            tokenizer.decode(TRANSLATION_TOKENS, skip_special_tokens=False),
            max_repeated_chars=6,
        ):
            cprint("LOOP DETECTED", "red", "on_yellow")
            PREVIOUS_FULL_WORDS = []
            time.sleep(5)
            finished = True
            wait = False
            break

    return TRANSLATION_TOKENS, complete_words_predicted, finished, wait


def predict_next_word_ja(src_text, TRANSLATION_TOKENS, source_finished=None):
    """Predict as many tokens as needed until a full word is formed"""
    global PREVIOUS_FULL_WORDS, WAIT_TOKEN_COUNT, GENERATION_CONFIG, WAIT_TOKENS_IN_SENTENCE

    finished = False
    wait = False

    prompt = make_prompt(src_text, TRANSLATION_TOKENS)
    encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    cprint(prompt, color="red")

    if source_finished:
        GENERATION_CONFIG.suppress_tokens = SUPPRESSED_TOKENS_FINISHED_INPUT + [2]
        GENERATION_CONFIG.repetition_penalty = 1.0
        GENERATION_CONFIG.max_new_tokens = 3
        GENERATION_CONFIG.forced_eos_token_id = None
        GENERATION_CONFIG.exponential_decay_length_penalty = (0, 2.0)
        cprint("source_finished", "blue", "on_red")
    else:
        GENERATION_CONFIG.suppress_tokens = SUPPRESSED_TOKENS_UNFINISHED_INPUT
        GENERATION_CONFIG.repetition_penalty = 1.0
        GENERATION_CONFIG.max_new_tokens = 3
        GENERATION_CONFIG.forced_eos_token_id = None
        GENERATION_CONFIG.exponential_decay_length_penalty = None

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            generation_config=GENERATION_CONFIG,
        )

    TRANSLATION_TOKENS += outputs[0][-GENERATION_CONFIG.max_new_tokens :].tolist()

    raw_string = tokenizer.decode(TRANSLATION_TOKENS, clean_up_tokenization_spaces=True)
    cprint(f"RAW string: {raw_string}", color="magenta")

    # cleanup unfinished Chinese characters
    raw_string = "".join([i for i in raw_string if i != "�"])

    # cleanup double commas
    raw_string = raw_string.replace("、、、", "、")
    raw_string = raw_string.replace("、、", "、")

    TRANSLATION_TOKENS = tokenizer(raw_string, add_special_tokens=False)["input_ids"]

    if 30267 in TRANSLATION_TOKENS:
        TRANSLATION_TOKENS = TRANSLATION_TOKENS[: (TRANSLATION_TOKENS.index(30267) + 1)]
        raw_string = tokenizer.decode(
            TRANSLATION_TOKENS, clean_up_tokenization_spaces=True
        )
        finished = True
        wait = False
        cprint("FINISHED", "red", "on_green", attrs=["bold"])

    _earliest_wait_token_idx = earliest_occurrence(TRANSLATION_TOKENS, WAIT_TOKEN_IDS)
    if _earliest_wait_token_idx:
        TRANSLATION_TOKENS = TRANSLATION_TOKENS[:_earliest_wait_token_idx]
        raw_string = tokenizer.decode(
            TRANSLATION_TOKENS, clean_up_tokenization_spaces=True
        )
        cprint("WAIT TOKEN DETECTED", "yellow", "on_green", attrs=["bold"])
        finished = False
        wait = True

    cprint(f"PROCESSED string: {raw_string}", color="cyan")

    new_content = get_new_content(PREVIOUS_FULL_WORDS, raw_string)
    print(f"NEW content: {new_content}")

    PREVIOUS_FULL_WORDS = copy.deepcopy(raw_string)

    if check_for_loop(raw_string):
        cprint("LOOP DETECTED", "red", "on_yellow")
        time.sleep(5)
        finished = True
        wait = False

    if len(new_content) == 0:
        wait = True

    if not wait:
        WAIT_TOKEN_COUNT = 0
    else:
        cprint("WAIT TOKEN generated", "red", "on_yellow")
        WAIT_TOKENS_IN_SENTENCE += 1
        WAIT_TOKEN_COUNT += 1
        if WAIT_TOKEN_COUNT > MAX_WAIT_TOKEN_COUNT:
            finished = True
            wait = False
            WAIT_TOKEN_COUNT = 0

    if finished:  # NOTE: clear previous words last thing before returning
        PREVIOUS_FULL_WORDS = ""

    return TRANSLATION_TOKENS, [new_content], finished, wait


def new_items(old_list, new_list):
    return [b for a, b in zip_longest(old_list, new_list) if not a]


PREDICT_FUNC = (
    predict_next_word_ja if TARGET_LANGUAGE == "Japanese" else predict_next_word
)
PREVIOUS_FULL_WORDS = [] if not TARGET_LANGUAGE == "Japanese" else ""
WAIT_TOKEN_COUNT = 0
WAIT_TOKENS_IN_SENTENCE = 0

print(
    f"SUPPRESSED_TOKENS_UNFINISHED_INPUT: {[REV_VOCAB[i] for i in SUPPRESSED_TOKENS_UNFINISHED_INPUT[-5:]]}"
)
print(
    f"SUPPRESSED_TOKENS_FINISHED_INPUT: {[REV_VOCAB[i] for i in SUPPRESSED_TOKENS_FINISHED_INPUT[-5:]]}"
)
print(f"WAIT_TOKEN_IDS: {[REV_VOCAB[i] for i in WAIT_TOKEN_IDS]}")
print(f"TARGET_LANGUAGE: {TARGET_LANGUAGE} | japanese_target: {japanese_target}")
print(f"PREDICT_FUNC: {PREDICT_FUNC}")


class ASR:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL_NAME
        ).to(DEVICE)
        self.asr_model.config.forced_decoder_ids = None
        self.srate = SRATE
        self.DEVICE = DEVICE

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if (
            source_finished
        ):  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array, source_finished):
        input_features = self.processor(
            audio_array,
            sampling_rate=self.srate,
            return_tensors="pt",
        ).input_features.to(self.DEVICE)
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        return self._postprocess(transcription, source_finished)


@entrypoint
class Agent(SpeechToTextAgent):

    TRANSLATION_TOKENS = []
    _japanese_target = japanese_target
    asr_model = ASR(ASR_MODEL_NAME, DEVICE, SRATE)
    input_words = []

    def policy(self):
        global PREVIOUS_FULL_WORDS, WAIT_TOKENS_IN_SENTENCE

        try:
            if len(self.TRANSLATION_TOKENS) == 0:
                self.start_time = time.time()

            # FIXME: you don't want the ASR model to re-recognize the sentence if it is already finished
            # recognize the whole audio segment (from start to now), return all the words from the start

            finished = False

            if not self.states.source_finished:
                if len(self.states.source) / SRATE <= MIN_READ_TIME:
                    return ReadAction()

            input_words = self.asr_model.recognize(
                self.states.source,
                self.states.source_finished,
            )

            # get the new words added on top of the previously recognized ones
            new_words = new_items(
                self.input_words, input_words
            )  # get new words from the full list of sentence words

            # # append only the first word
            # if len(new_words) > 1:
            #     print('More than one new word.')
            #     print('Discarding all but the first new word...')
            #     new_words = new_words[0]

            cprint(f"Old words: {self.input_words}", "blue", "on_white", attrs=["bold"])
            cprint(f"ASR output: {input_words}", "grey", "on_white", attrs=["bold"])
            cprint(f"New words: {new_words}", "green", "on_white", attrs=["bold"])

            if len(new_words) > 0:
                self.input_words.append(new_words[0])  # append only the first word
            cprint(
                f"Updated words: {self.input_words}", "red", "on_white", attrs=["bold"]
            )

            # if the ASR returned no new words, read in more
            if (len(new_words) == 0) or (
                (len(self.input_words) < WAIT_K) and not self.states.source_finished
            ):
                if not self.states.source_finished:
                    cprint(
                        "resetting WAIT_TOKENS_IN_SENTENCE, returning ReadAction()",
                        color="magenta",
                        attrs=["bold"],
                    )
                    WAIT_TOKENS_IN_SENTENCE = 0
                    return ReadAction()

            src_text = " ".join(
                self.input_words
            )  # join source words into text (words are with punctuation)

            # predict as many tokens as needed for one whole word, finished=True if EOS or FS has been generated
            self.TRANSLATION_TOKENS, complete_words_predicted, finished, wait = (
                PREDICT_FUNC(
                    src_text,
                    self.TRANSLATION_TOKENS,
                    source_finished=self.states.source_finished,
                )
            )

            # stop if all the source words have been revealed, but target is 2x the source
            if self.states.source_finished:
                ratio = len(complete_words_predicted) / (0.1 + len(self.input_words))
                cprint(
                    f"wait: {wait} | finished: {finished} | ratio: {ratio}",
                    color="green",
                )
                if ratio > 3:
                    PREVIOUS_FULL_WORDS = []  # clear previous words
                    finished = True
                    wait = False
                    print()
                    cprint("Forcefully terminated the sentence", color="yellow")
                    print()

            if finished:
                self.WAIT_TOKENS_IN_SENTENCE = WAIT_TOKENS_IN_SENTENCE
                self.TRANSLATION_TOKENS = []  # clear the list of translation tokens
                with open(ONLINE_TRANSLATION_PATH, "a", encoding="utf-8") as f:
                    if TARGET_LANGUAGE != "Japanese":
                        f.write(" ".join(complete_words_predicted) + "\n")
                    else:
                        f.write(
                            "".join(self.states.target)
                            + complete_words_predicted[-1]
                            + "\n"
                        )

                # with open(f"timer_{ONLINE_TRANSLATION_PATH}", 'a', encoding='utf-8') as f:
                #     f.write(f'{(time.time() - self.start_time):.2f}\n')

            if TARGET_LANGUAGE != "Japanese":
                cprint(
                    f'{src_text} : {" ".join(complete_words_predicted)}', color="blue"
                )
            else:
                cprint(f'{src_text} : {"".join(self.states.target)}', color="blue")
            cprint(
                f"WAIT: {wait} FINISHED: {finished} SRC_FINISHED: {self.states.source_finished} len_states_source:  {len(self.states.source)}",
                color="magenta",
            )

            if finished and not wait:
                with open(ASR_PATH, "a", encoding="utf-8") as f:
                    f.write(" ".join(self.input_words) + "\n")
                self.input_words = []

            if not wait:
                try:
                    return WriteAction(complete_words_predicted[-1], finished=finished)
                except:
                    cprint(complete_words_predicted, color="cyan")
                    return ReadAction()
            else:
                return ReadAction()

        except Exception as e:
            self.TRANSLATION_TOKENS = []  # class property
            PREVIOUS_FULL_WORDS = []  # global var
            WAIT_TOKEN_COUNT = 0  # gloabl var
            WAIT_TOKENS_IN_SENTENCE = 0  # global var
            self.WAIT_TOKENS_IN_SENTENCE = 0
            return WriteAction(f"{e}", finished=True)
