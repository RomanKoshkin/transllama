from transformers import GenerationConfig, AutoTokenizer
import re, time
from termcolor import cprint

self_name = __file__.split('/')[-1].split('.')[0]  #__name__.split(".")[-1]


TARGET_LANGUAGE = "German"
MODEL_NAME = "nightdude/transllama-70-en-de"
cprint(f"loading the MERGED {MODEL_NAME} model", 'black', 'on_yellow', attrs=['bold'])
ADAPTER_NAME = f"nightdude/{self_name}"  # path to the LoRA adapter (will be strapped onto the base model)
TRAIN_DS_PATH = '../data/mustc2_ende_aligned_dict_4000_train.json'
VALID_DS_PATH = '../data/mustc2_ende_aligned_dict_4000_valid.json'

##################################################################################################
########################## PROMPT CONFIGS ########################################################
##################################################################################################

WAIT_TOKEN = "▁▁"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "\n<<SYS>>\n\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = f"You are a professional conference interpreter. Given an English text you translate it into {TARGET_LANGUAGE} as accurately and as concisely as possible, NEVER adding comments of your own. You output translation when the information available in the source is unambiguous, otherwise you output the null token (\"{WAIT_TOKEN}\"), not flanked by anything else. It's important that you get this right."

##################################################################################################
########################## FINE-TUNING CONFIGS ###################################################
##################################################################################################

LOGWEIGHTS = True

# LORA_CONFIG = LoraConfig(  # LoraConfig is a type of PeftConfig that you must pass to get a PEFT model
#     r=16,
#     lora_alpha=32,  # bottleneck dimension in the adapter
#     target_modules=['gate_proj', 'k_proj', 'o_proj', 'up_proj', 'v_proj', 'down_proj', 'q_proj'],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM")

# TRAINING_ARGS = TrainingArguments(
#     evaluation_strategy="steps",
#     load_best_model_at_end=False,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     eval_steps=10,
#     per_device_train_batch_size=25,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,
#     learning_rate=6e-5,
#     fp16=True,
#     save_total_limit=1,
#     logging_steps=1,
#     report_to=None,  # disable WandB and TensorBoard
#     output_dir=f"../experiments/{self_name}",
#     optim="paged_adamw_32bit",
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.05,
#     remove_unused_columns=False  # with SamplingDataCollatorForLanguageModeling don't remove anything!
# )

# BNB_CONFIG = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

##################################################################################################
########################## GENERATION CONFIGS ####################################################
##################################################################################################

GENERATION_CONFIG = GenerationConfig()
GENERATION_CONFIG.max_new_tokens = 1
# GENERATION_CONFIG.temperature = 1.0 # temp will be overridden by top_p
GENERATION_CONFIG.top_p = 0.7
GENERATION_CONFIG.num_return_sequences = 1
GENERATION_CONFIG.bos_token_id = 1
GENERATION_CONFIG.pad_token_id = 0
GENERATION_CONFIG.eos_token_id = 2
GENERATION_CONFIG.repetition_penalty = 1.0  # NOTE:!!!!!!
GENERATION_CONFIG.do_sample = False

PREDICT_TOKEN_BY_TOKEN = False  # keep reading even if the output words are not finished yet
WAIT_K = 5
NUM_BEAMS = 1
BEAM_DEPTH = 1
MAX_WAIT_TOKEN_COUNT = 10  # NOTE: !!!!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto")

# NOTE: set the pad token to UNK (0), because it has a low probability and is not masked in the labels
tokenizer.pad_token = '<unk>'
cprint(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}", color='blue')
time.sleep(3)

WAIT_TOKEN_ID = tokenizer(WAIT_TOKEN, add_special_tokens=False)['input_ids'][-1]
REV_VOCAB = {v: k for k, v in tokenizer.vocab.items()}
WAIT_TOKEN_IDS = [259, WAIT_TOKEN_ID]

japanese_target = True if TARGET_LANGUAGE == 'Japanese' else False
#NOTE: don't suppress 29871 for de, ru, otherwise you'll get unfinished sentences
JUNK_EXPR = r'[\'~@#\$%&\+=\"\]\[\(\)\|_\*\^<>\{\}\?!\\\/]' if not japanese_target else r'[「」\.\'~@#\$%&\+=\"\]\[\(\)\|_\*\^<>\{\}\?!\\\/]'

contains_latin_chars = lambda v: bool(re.search('[a-zA-Z]', v)) if TARGET_LANGUAGE in ['Russian', 'Japanese'] else False

JUNK = []
for ID, v in REV_VOCAB.items():
    if bool(re.search(JUNK_EXPR, v)) or contains_latin_chars(v):
        JUNK.append(ID)
[JUNK.remove(i) for i in ([0, 1, 2])]  # <unk> <eos> and <bos> are not junk
if japanese_target:
    JUNK.append(29871)
    JUNK.append(13)
    [JUNK.remove(i) for i in range(3, 259)]  # BPEs for Chinese characters

cprint('EXPERIMENTAL: SUPPRESSING NUMBERS', 'grey', 'on_blue')
time.sleep(4)
for ID, v in REV_VOCAB.items():
    if bool(re.search(r'[0123456789]', v)):
        if ID > 20000:
            JUNK.append(ID)
JUNK += [30064, 2880, 30725, 30706, 31188, 30160, 29908, 30318]  # NOTE: experimental ('·■✓↵″"►' tokens)
JUNK += [539, 30245, 869, 6317, 2023, 25285, 11296, 856, 847, 29914, 30098, 10266]  # back arrow
JUNK = list(set(JUNK))

if japanese_target:
    SUPPRESSED_TOKENS_UNFINISHED_INPUT = JUNK + [2, 30267, 30882]  # EOS, full stop, question mark
else:
    SUPPRESSED_TOKENS_UNFINISHED_INPUT = JUNK + [2, 29889, 29973, 1577]  # EOS, full stop, question mark
SUPPRESSED_TOKENS_FINISHED_INPUT = JUNK + [WAIT_TOKEN_ID, 259] + [30330]  # wait tokens

assert TARGET_LANGUAGE in [
    'Japanese', 'Russian', 'German'
], f"{TARGET_LANGUAGE} is not supported as a target language. Choose from Japanese, Russian or German."
