import os, sys
from dotenv import load_dotenv
from termcolor import cprint

load_dotenv('../.env', override=True)  # load API keys into
PATH_ROOT = os.getenv("PATH_ROOT")
cprint(PATH_ROOT, 'black', 'on_yellow')

import numpy as np
from typing import Tuple

LANG = sys.argv[1]
assert LANG in ['ru', 'de'], "WRONG LANGUAGE"
with open(os.path.join(PATH_ROOT, f'evaluation/SOURCES/src_ted_new_tst_100.{LANG}'), 'r') as f:
    a = [i[:-1].strip() for i in f.readlines()]
with open(os.path.join(PATH_ROOT, f'evaluation/SOURCES/src_ted_new_tst_100.{LANG}.txt'), 'r') as f:
    s = [i[:-1].replace('\xa0', ' ').strip() for i in f.readlines()]
with open(os.path.join(PATH_ROOT, f'evaluation/OFFLINE_TARGETS/tgt_ted_new_tst_100.{LANG}'), 'r') as f:
    t = [i[:-1].replace('\xa0', ' ').strip() for i in f.readlines()]
assert len(s) == len(t) == len(a)

A = [A for A in zip(a, s, t)]


def resample_and_save(A: Tuple[str, str]) -> None:
    idx = np.random.choice(len(A), len(A), replace=True)
    uniques, counts = np.unique(idx, return_counts=True)
    print(f'Uniques: {len(uniques)}, max_repeats: {max(counts)}')
    resampled = [A[i] for i in idx]
    a_res = [i[0] for i in resampled]
    s_res = [i[1] for i in resampled]
    t_res = [i[2] for i in resampled]

    with open('resampled_src_new_ted_audio.txt', 'w') as f:
        f.writelines([i + '\n' for i in a_res])
    with open('resampled_src_new_ted_text.txt', 'w') as f:
        f.writelines([i + '\n' for i in s_res])
    with open('resampled_tgt_new_ted.txt', 'w') as f:
        f.writelines([i + '\n' for i in t_res])


resample_and_save(A)