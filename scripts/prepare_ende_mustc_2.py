import sys, os, json
from tqdm import tqdm
sys.path.append('../')

from dotenv import load_dotenv
load_dotenv('../.env', override=True)  # load API keys into


from functools import partial
import json

from simalign import SentenceAligner # uses bert_base_multilingual_cased from HuggingFace

from datasets import load_dataset, Dataset
from scripts.align_mustc import alignVerboseSourceToConciseTargetNEW
from utils.llm_utils import cleanup



dataset = load_dataset("enimai/MuST-C-de")
dataset = dataset.rename_columns({'en':'en', 'de':'ru'})


# make an instance of our model. Specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai", device='cuda:0')

D_tr =  dataset['train'].shuffle(seed=0).to_list()
D_val = dataset['validation'].shuffle(seed=0).to_list()


# minlen, maxlen is 9, 70 (hard-coded in utils.llm_utils.cleanup)
N = 4000 # we'll subset 1000 random samples
SHIFT_INTO_FUTURE = 0 # originally no shift


cleanup_long = partial(cleanup, min_num_words_in_source=1)
cleanup_short = partial(cleanup, min_num_words_in_source=9)

D_tr = list(filter(None, map(cleanup_long, D_tr))) # map with cleanpu and filter items that were bad (two or more sentences)
D_val = list(filter(None, map(cleanup_short, D_val))) # map with cleanpu and filter items that were bad (two or more sentences)

D_train = D_tr[:N] # we select 1000 random samples AFTER rejecting bad samples and getting clean ones
D_valid = D_val[:100] # we select 10% of the validattion set samples AFTER rejecting bad samples and getting clean ones
aligned_sentences_train = alignVerboseSourceToConciseTargetNEW(myaligner, D_train, shift_into_future=SHIFT_INTO_FUTURE)
aligned_sentences_valid = alignVerboseSourceToConciseTargetNEW(myaligner, D_valid, shift_into_future=SHIFT_INTO_FUTURE)

# save a List[Dict[src: trg]], in which src and trg are aligned words
with open(f'../data/mustc2_ende_aligned_dict_{N}_train.json', 'w') as f:
    f.write(json.dumps(aligned_sentences_train, ensure_ascii=False))
    
with open(f'../data/mustc2_ende_aligned_dict_{N}_valid.json', 'w') as f:
    f.write(json.dumps(aligned_sentences_valid, ensure_ascii=False))