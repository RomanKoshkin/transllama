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

# PATH_TO_UNALIGNED_PAIRS = '../data/dat/for_alignment_gpt3'
# PATH_TO_UNALIGNED_PAIRS = '../data/dat/for_alignment_gpt4'
PATH_TO_UNALIGNED_PAIRS = '../data/dat/for_alignment_tgt_by_gpt4'
PATH_TO_ALIGNED_PAIRS = '../data/dat/aligned'


@njit
def align(edges):
    """ shifts the nodes of the right side of the bipartite graph (SRC-TGT) """
    for i in range(edges.shape[0]):
        s_node, t_node = edges[i][0], edges[i][1]
        if s_node > t_node:
            for j in range(i, edges.shape[0]):
                edges[j, 1] += 1
            return i, edges
    return None, edges


def alignVerboseSourceToConciseTarget(myaligner: SentenceAligner, D: list) -> list:
    """ takes a list of dicts: {'source': <SRC>, 'target': <CONCISE_TGT>} """
    A = []
    for it in tqdm(D):
        S = nltk.word_tokenize(it['source'])
        T = nltk.word_tokenize(it['target'])

        alignments = myaligner.get_word_aligns(S, T)
        alignments['itermax'] = alignments['itermax'][1:]  # ????

        edges = np.array(alignments['itermax'])

        while True:
            i, edges = align(edges)
            if i is None:
                break
            else:
                T.insert(i, WAIT_TOKEN)
        length_mismatch = len(T) - len(S)
        if length_mismatch > 0:
            for i in range(length_mismatch):
                S.append(WAIT_TOKEN)
        edges = [tuple(it) for it in edges]

        A.append({'source': S, 'target': T})
    return A


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
def alignVerboseSourceToConciseTargetNEW(myaligner: SentenceAligner, D: list, shift_into_future: int = 0) -> list:
    """ takes a list of dicts: {'source': <SRC>, 'target': <CONCISE_TGT>} """
    A = []
    for it in tqdm(D):
        S = nltk.word_tokenize(it['source'])
        T = nltk.word_tokenize(it['target'])

        alignments = myaligner.get_word_aligns(S, T)

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


# instantiate the model. Specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(
    model="bert",
    token_type="bpe",
    matching_methods="mai",
)

# align one text at a time
for fname in os.listdir(PATH_TO_UNALIGNED_PAIRS):
    if fname.endswith('.json'):
        TEXT_ID = int(fname.split('.')[0])
    else:
        continue

    # read in the source-target pairs (in JSON format) into a list of dicts
    with open(f'{PATH_TO_UNALIGNED_PAIRS}/{TEXT_ID}.json', 'r') as f:
        D = json.loads(f.read())

    aligned_sentences = alignVerboseSourceToConciseTargetNEW(myaligner, D)

    # sanity check
    for d in aligned_sentences:
        assert len(d['source']) == len(d['target']), "source and target not equal after alignment"

    # save the aligned sentences
    for sid, sentence in enumerate(aligned_sentences):
        with open(f'{PATH_TO_ALIGNED_PAIRS}/{TEXT_ID:03d}_{sid:03d}.json', 'w') as f:
            f.write(pd.DataFrame(sentence).to_json(
                force_ascii=False,
                orient='records',
            ))
