import os, sys
from dotenv import load_dotenv

load_dotenv('../.env', override=True)  # load API keys into
sys.path.append('../')

import nltk, json
import numpy as np
import pandas as pd
from simalign import SentenceAligner  # uses bert_base_multilingual_cased from HuggingFace
from numba import njit
from tqdm import tqdm
from utils.constants import WAIT_TOKEN
from datasets import load_dataset
from utils.llm_utils import remove_parentheses, remove_multiple_spaces
from janome.tokenizer import Tokenizer
from termcolor import cprint

PATH_TO_ALIGNED_PAIRS = '../data/dat/aligned'


def fix_alignment(alignment):
    """"
    fix the alignments by removing edges between many source nodes and one target node 
    (many-to-one and one-to-many) and only keeping the last
    """
    seen = set()
    _result = []
    for item in reversed(alignment):
        if item[1] not in seen:
            _result.append(item)
            seen.add(item[1])
    _result = list(reversed(_result))

    seen = set()
    result_ = []
    for item in reversed(_result):
        if item[0] not in seen:
            result_.append(item)
            seen.add(item[0])

    return list(reversed(result_))


# new alignment function
def alignVerboseSourceToConciseTargetNEW(
    myaligner: SentenceAligner,
    D: list,
    japanese_target: bool = False,
    shift_into_future: int = 0,
) -> list:
    """ takes a list of dicts: {'source': <SRC>, 'target': <CONCISE_TGT>} """
    A = []
    ja_tokenizer = Tokenizer()
    pbar = tqdm(D)
    for i, it in enumerate(pbar):
        pbar.set_description(f'Aligning item {i}')

        try:
            S = nltk.word_tokenize(it['en'])
            if japanese_target:
                T = [token.surface for token in ja_tokenizer.tokenize(it['ru'])]
            else:
                T = nltk.word_tokenize(it['ru'])
            alignments = myaligner.get_word_aligns(S, T)
        except Exception as e:
            cprint(f'Problem aligning {it}. Exception: {e}', color='yellow')
            continue

        # remove one-to-many, many-to-one connections, keeping only the latest
        alignments['itermax'] = fix_alignment(alignments['itermax'])

        edges_np = np.array(alignments['itermax'])
        for i in range(edges_np.shape[0]):
            while edges_np[i, 1] < edges_np[i, 0] + shift_into_future:
                target_node_id = edges_np[i, 1]
                edges_np[i:, 1] += 1
                T.insert(target_node_id, WAIT_TOKEN)  # just before

        length_mismatch = len(T) - len(S)
        if length_mismatch > 0:
            for i in range(length_mismatch):
                S.append(WAIT_TOKEN)

        A.append({'source': S, 'target': T})
    return A


if __name__ == '__main__':

    # instantiate the model. Specify the embedding model and all alignment settings in the constructor.
    myaligner = SentenceAligner(
        model="bert",
        token_type="bpe",
        matching_methods="mai",
    )

    # read in the source-target pairs (in JSON format) into a list of dicts
    D = load_dataset("enimai/MuST-C-ru")['train'].to_list()[:10]

    # align all the sentences
    aligned_sentences = alignVerboseSourceToConciseTargetNEW(myaligner, D)

    # sanity check
    for d in aligned_sentences:
        assert len(d['source']) == len(d['target']), "source and target not equal after alignment"

    # save the aligned sentences
    for sid, sentence in enumerate(aligned_sentences):
        with open(f'{PATH_TO_ALIGNED_PAIRS}/mustc_train_{sid:04d}.json', 'w') as f:
            f.write(pd.DataFrame(sentence).to_json(
                force_ascii=False,
                orient='records',
            ))
