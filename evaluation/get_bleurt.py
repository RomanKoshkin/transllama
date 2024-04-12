import os, sys, argparse
from bleurt import score
from termcolor import cprint
""""
Expects the targets to be sentences, each on its own line, terminated by a period and caret return
"""

from dotenv import load_dotenv

load_dotenv('../.env', override=True)  # load API keys into
sys.path.append('../')

parser = argparse.ArgumentParser(description="Bleurt")
# parser.add_argument('--on', action='store_true', help="use part of the data to speed things up.")
parser.add_argument("--online", type=str, default=None)
parser.add_argument("--offline", type=str, default=None)
args = parser.parse_args()

with open(f'{args.online}', 'r') as f:
    online_target = f.readlines()
    num_sentences_in_online = len(online_target)
    online_target = " ".join([i[:-1] for i in online_target])

with open(f'{args.offline}', 'r') as f:
    offline_target = f.readlines()
    offline_target = " ".join([i[:-1] for i in offline_target[:num_sentences_in_online]])

# Calculate BLEURT score
scorer = score.BleurtScorer(f'{os.environ["HF_HOME"]}/BLEURT-20')
bleurt_scores = scorer.score(references=[offline_target], candidates=[online_target])
cprint(bleurt_scores, color='yellow')
