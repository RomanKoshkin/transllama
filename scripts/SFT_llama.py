import torch, os, sys, argparse, json, time
from dotenv import load_dotenv
from importlib import import_module
from termcolor import cprint

sys.path.append('../')

load_dotenv('../.env', override=True)
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description="fine-tune given a config_id")
parser.add_argument("--config_id", type=int, default=-1)
parser.add_argument("--device_id", type=int, default=-1)
parser.add_argument("--mask_prompt", action='store_true')
args = parser.parse_args()
assert (args.config_id >= 0), 'Required argument: config_id missing'
assert (args.device_id >= 0), 'Required argument: device_id missing'
cprint(f"args.mask_prompt: {bool(args.mask_prompt)}", 'grey', 'on_red')
time.sleep(4)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)  # NOTE: order is important
DEVICE = torch.cuda.current_device()

# import CONFIG
sys.path.append('../configs')
module_name = f"config_{args.config_id}"
module = import_module(module_name)
B_INST = getattr(module, 'B_INST')
E_INST = getattr(module, 'E_INST')
B_SYS = getattr(module, 'B_SYS')
E_SYS = getattr(module, 'E_SYS')
DEFAULT_SYSTEM_PROMPT = getattr(module, 'DEFAULT_SYSTEM_PROMPT')
BNB_CONFIG = getattr(module, 'BNB_CONFIG')
MODEL_NAME = getattr(module, 'MODEL_NAME')
ADAPTER_NAME = getattr(module, 'ADAPTER_NAME')
TRAIN_DS_PATH = getattr(module, 'TRAIN_DS_PATH')
VALID_DS_PATH = getattr(module, 'VALID_DS_PATH')
LORA_CONFIG = getattr(module, 'LORA_CONFIG')
TRAINING_ARGS = getattr(module, 'TRAINING_ARGS')
LOGWEIGHTS = getattr(module, 'LOGWEIGHTS')
WAIT_TOKEN = getattr(module, 'WAIT_TOKEN')
try:
    japanese_target = getattr(module, 'japanese_target')
except:
    japanese_target = False

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training

from utils.custom_collator import SamplingDataCollatorForLanguageModelingJA
from utils.llm_utils import print_trainable_parameters
from utils.datasets import makeCLMdatasetFromListOfDictsOfAlignedWords

print(f"device: {torch.cuda.current_device()}")
print(f'device: {os.environ["CUDA_VISIBLE_DEVICES"]}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

with open(TRAIN_DS_PATH, 'r') as f:
    D = json.loads(f.read())
train_ds = makeCLMdatasetFromListOfDictsOfAlignedWords(tokenizer, D, japanese_target=japanese_target)

with open(VALID_DS_PATH, 'r') as f:
    D = json.loads(f.read())
valid_ds = makeCLMdatasetFromListOfDictsOfAlignedWords(tokenizer, D, japanese_target=japanese_target)

torch.cuda.set_device(DEVICE)

# DEFINE THE MODEL
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,  #"auto",
    trust_remote_code=True,
    quantization_config=BNB_CONFIG,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)
print_trainable_parameters(model)

TRAINING_ARGS.run_name = module_name
trainer = Trainer(
    model=model,
    train_dataset=train_ds,  # dataset here must be a list of dicts
    eval_dataset=valid_ds,
    args=TRAINING_ARGS,
    data_collator=SamplingDataCollatorForLanguageModelingJA(
        tokenizer,
        WAIT_TOKEN,
        mlm=False,
        logweights=LOGWEIGHTS,
        japanese_target=japanese_target,
        mask_prompt=bool(args.mask_prompt),
    )  # Data collators are objects that will form a batch by using a list of dataset elements as input
)
model.config.use_cache = False
trainer.train()

model.push_to_hub(module_name)  # ADAPTER_NAME and `config_<ID>` are the same

# best_step = trainer.state.best_step
# TMP_PATH = '../.tmp'
# if not os.path.exists(TMP_PATH):
#     os.makedirs(TMP_PATH)

# with open(f'{TMP_PATH}/training_notes_{module_name}.json', 'w') as f:
#     json.dump({"best_step": best_step}, f)

# with open('training_notes.json', 'w') as f:
#     json.dump({"best_step": best_step}, f)

# api = HfApi()
# api.upload_file(
#     path_or_fileobj=f'training_notes_{module_name}.json',
#     path_in_repo=f'training_notes_{module_name}.json',
#     repo_id=f"nightdude/{module_name}",
#     repo_type="model",
# )