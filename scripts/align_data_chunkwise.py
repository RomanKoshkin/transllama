import sys, json
from dotenv import load_dotenv

sys.path.append('../')

load_dotenv('../.env', override=True)

from functools import partial
from simalign import SentenceAligner  # uses bert_base_multilingual_cased from HuggingFace

from datasets import load_dataset, Dataset
from scripts.align_mustc import alignVerboseSourceToConciseTargetNEW
from scripts.align_mustc import fix_alignment
from utils.llm_utils import cleanup

FROM = int(sys.argv[1])
TO = int(sys.argv[2])
DEVICE = f'cuda:{int(sys.argv[3])}'

dataset = load_dataset("enimai/MuST-C-de")
dataset = dataset.rename_columns({'en': 'en', 'de': 'ru'})

# making an instance of our model. Specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai", device=DEVICE)

D_tr = dataset['train'].shuffle(seed=0).to_list()
D_val = dataset['validation'].shuffle(seed=0).to_list()

cleanup_long = partial(cleanup, min_num_words_in_source=1)
cleanup_short = partial(cleanup, min_num_words_in_source=9)

# map with cleanup and filter items that were bad (two or more sentences)
D_tr = list(filter(None, map(cleanup_long, D_tr)))

D_train = D_tr[FROM:TO]  # we select 1000 random samples AFTER rejecting bad samples and getting clean ones

aligned_sentences_train = alignVerboseSourceToConciseTargetNEW(myaligner, D_train)

# save a List[Dict[src: trg]], in which src and trg are aligned words
with open(f'../data/mustc2_ende_aligned_dict_{FROM}_{TO}_train.json', 'w') as f:
    f.write(json.dumps(aligned_sentences_train, ensure_ascii=False))
