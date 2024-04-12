# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from termcolor import cprint
import sys, os, json, time
import logging
from simuleval import options
from simuleval.utils.agent import build_system_args
from simuleval.utils.slurm import submit_slurm_job
from simuleval.utils.arguments import check_argument, cli_argument_list
from simuleval.utils import EVALUATION_SYSTEM_LIST
from simuleval.evaluator import (
    build_evaluator,
    build_remote_evaluator,
    SentenceLevelEvaluator,
)
from simuleval.agents.service import start_agent_service
from simuleval.agents import GenericAgent

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)

logger = logging.getLogger("simuleval.cli")


def main():

    xxx = dict()
    xxx['ZERO_SHOT'] = bool(check_argument("zero_shot"))
    cprint(xxx, color='yellow')
    time.sleep(5)

    if check_argument("remote_eval"):
        remote_evaluate()
        return

    if check_argument("score_only"):
        scoring()
        return

    if check_argument("slurm"):
        submit_slurm_job()
        return

    system, args = build_system_args()  # here the agent gets built. takes optional `config_dict`

    if check_argument("standalone"):
        start_agent_service(system)
        return

    # build evaluator
    evaluator = build_evaluator(args)  # will call the system.policy method from its __call__ method
    # evaluate system. This will call the __call__ method, which will iterate over sentences, 1 by 1, calc metrics
    results, predictions, references, proportion_of_wait_tokens = evaluator(system)

    # get BLEURT score
    from bleurt import score
    scorer = score.BleurtScorer(f'{os.environ["HF_HOME"]}/BLEURT-20')
    bleurt_scores = scorer.score(references=[references], candidates=[predictions])
    results['BLEURT'] = round(float(bleurt_scores[0]), 3)
    results['CONFIG_ID'] = int(check_argument("config_id"))
    results['NUM_BEAMS'] = int(check_argument("num_beams"))
    results['BEAM_DEPTH'] = int(check_argument("beam_depth"))
    results['ZERO_SHOT'] = bool(check_argument("zero_shot"))
    results['WAIT_K'] = int(check_argument("wait_k"))
    results['WAIT_T_PROR'] = round(proportion_of_wait_tokens, 3)

    parser_aux = argparse.ArgumentParser()
    parser_aux.add_argument("--agent", type=str, default="")
    parser_aux.add_argument("--config_id", type=int, default=-1)
    parser_aux.add_argument("--min_lag", type=int, default=-1)
    parser_aux.add_argument("--asr_model", type=str, default="")
    parser_aux.add_argument("--source-segment-size", type=int, default=-1)
    parser_aux.add_argument("--source", type=str, default="")
    parser_aux.add_argument("--target", type=str, default="")
    aux_args, _ = parser_aux.parse_known_args(cli_argument_list(None))

    if check_argument('agent').startswith('s2t'):
        results['MIN_LAG_WRDS'] = aux_args.min_lag
        results['ASR_MODEL'] = aux_args.asr_model
        results['SRC_SEG_MS'] = aux_args.source_segment_size
        results['SRC'] = aux_args.source
        results['TGT'] = aux_args.target
        with open('S2T_METRICS.json', 'a') as f:
            f.write(json.dumps(results.to_dict(orient='records')[0]) + "\n")
        cprint(results.to_string(), color='red')
    else:
        with open('METRICS.json', 'a') as f:
            f.write(json.dumps(results.to_dict(orient='records')[0]) + "\n")
        cprint(results.to_string(), color='red')


def evaluate(system_class: GenericAgent, config_dict: dict = {}):
    EVALUATION_SYSTEM_LIST.append(system_class)

    if check_argument("slurm", config_dict):
        submit_slurm_job(config_dict)
        return

    system, args = build_system_args(config_dict)

    # build evaluator
    evaluator = build_evaluator(args)
    # evaluate system
    evaluator(system)


def scoring():
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser)
    options.add_dataloader_args(parser)
    args = parser.parse_args()
    evaluator = SentenceLevelEvaluator.from_args(args)
    print(evaluator.results)


def remote_evaluate():
    # build evaluator
    parser = options.general_parser()
    options.add_dataloader_args(parser)
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser)
    args = parser.parse_args()
    evaluator = build_remote_evaluator(args)

    # evaluate system
    evaluator.remote_eval()


if __name__ == "__main__":
    main()
