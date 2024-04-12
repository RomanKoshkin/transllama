# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas
import numpy as np
import os, sys
import numbers
from argparse import Namespace
from typing import Dict, Generator, Optional
from .scorers import get_scorer_class
from .scorers.latency_scorer import LatencyScorer
from .scorers.quality_scorer import QualityScorer

from .instance import INSTANCE_TYPE_DICT, LogInstance
import yaml
import logging
import json
from tqdm import tqdm
from pathlib import Path
from simuleval.data.dataloader import GenericDataloader, build_dataloader
from termcolor import cprint

sys.path.append('../')

from utils.llm_utils import fix_spaces_around_punctuation, contains_japanese

logger = logging.getLogger("simuleval.sentence_level_evaluator")


class SentenceLevelEvaluator(object):
    """
    Sentence Level evaluator. It iterates over sentence pairs and run evaluation.


    .. code-block:: python

        for instance in self.maybe_tqdm(self.instances.values()):
            agent.reset()
            while not instance.finish_prediction:
                input_segment = instance.send_source(self.source_segment_size)
                output_segment = agent.pushpop(input_segment)
                instance.receive_prediction(output_segment)


    Attributes:
        instances: collections of sentence pairs. Instances also keep track of delays.
        latency_scorers (List[~simuleval.scorers.latency_scorer.LatencyScorer]): Scorers for latency evaluation.
        quality_scorers (List[~simuleval.scorers.latency_scorer.QualityScorer]): Scorers for quality evaluation.
        output: output directory

    Evaluator related command line arguments:

    .. argparse::
        :ref: simuleval.options.add_evaluator_args
        :passparser:
        :prog:
    """

    def __init__(
        self,
        dataloader: Optional[GenericDataloader],
        quality_scorers: Dict[str, QualityScorer],
        latency_scorers: Dict[str, LatencyScorer],
        args: Namespace,
    ) -> None:
        self.dataloader = dataloader
        self.quality_scorers = quality_scorers
        self.latency_scorers = latency_scorers
        self.instances = {}

        self.args = args
        self.output = Path(args.output) if args.output else None
        self.score_only = args.score_only
        self.source_segment_size = getattr(args, "source_segment_size", 1)
        self.source_type = getattr(args, "source_type", None)
        self.target_type = getattr(args, "target_type", None)

        if (self.source_type is None and self.target_type is None and self.output is not None):
            with open(self.output / "config.yaml") as f:
                configs = yaml.safe_load(f)
                self.source_type = configs["source_type"]
                self.target_type = configs["target_type"]

        assert self.source_type
        assert self.target_type

        if self.output is not None:
            os.makedirs(self.output, exist_ok=True)
            with open(self.output / "config.yaml", "w") as f:
                yaml.dump(
                    {
                        "source_type": self.source_type,
                        "target_type": self.source_type
                    },
                    f,
                    default_flow_style=False,
                )

        self.instance_class = INSTANCE_TYPE_DICT[f"{self.source_type}-{self.target_type}"]
        self.start_index = getattr(args, "start_index", 0)
        self.end_index = getattr(args, "end_index", -1)

        if not self.score_only:
            if self.output:
                if (self.args.continue_unfinished and (self.output / "instances.log").exists()):
                    with open(self.output / "instances.log", "r") as f:
                        line = None
                        for line in f:  # noqa
                            pass
                        if line is not None:
                            last_info = json.loads(line.strip())
                            self.start_index = last_info["index"] + 1
                else:
                    self.output.mkdir(exist_ok=True, parents=True)
                    open(self.output / "instances.log", "w").close()
            if self.end_index < 0:
                assert self.dataloader is not None
                self.end_index = len(self.dataloader)

        self.build_instances()  # NOTE: instances are sentences

        if not self.args.no_progress_bar and not self.score_only:
            self.instance_iterator = tqdm(
                self.instances.values(),
                initial=self.start_index,
                total=len(self.instances.values()),
            )
        else:
            self.instance_iterator = self.instances.values()

    def write_log(self, instance):
        if self.output is not None:
            with open(self.output / "instances.log", "a") as f:
                f.write(json.dumps(instance.summarize()) + "\n")

    def build_instances(self):
        if self.score_only:
            self.build_instances_from_log()
        else:
            self.build_instances_from_dataloader()

    def build_instances_from_log(self):
        self.instances = {}
        if self.output is not None:
            with open(self.output / "instances.log", "r") as f:
                for line in f:
                    instance = LogInstance(line.strip())
                    self.instances[instance.index] = instance

    def build_instances_from_dataloader(self):
        for i in self.get_indices():
            self.instances[i] = self.instance_class(i, self.dataloader, self.args)  # NOTE: instances are sentences

    def __len__(self) -> int:
        return self.end_index - self.start_index

    def get_indices(self) -> Generator:
        if self.end_index < 0:
            self.end_index = max(self.instances.keys()) + 1

        if self.start_index > self.end_index:
            return []

        for index in range(self.start_index, self.end_index):
            yield index

    @property
    def quality(self) -> Dict[str, float]:  # instances are source/onli_tgt/offl_tgt triplets (objects)
        return {name: scorer(self.instances) for name, scorer in self.quality_scorers.items()}  # pass to __call__

    @property
    def latency(self) -> Dict[str, float]:
        return {name: scorer(self.instances) for name, scorer in self.latency_scorers.items()}

    @property
    def results(self):
        scores = {**self.quality, **self.latency}
        new_scores = {}
        for name, value in scores.items():
            if isinstance(value, numbers.Number):
                value = round(value, 3)
            new_scores[name] = [value]

        df = pandas.DataFrame(new_scores)
        return df

    def dump_results(self) -> None:
        results = self.results
        if self.output:
            results.to_csv(self.output / "scores.tsv", sep="\t", index=False)

        logger.info("Results:")
        print(results.to_string(index=False))
        return results

    def dump_metrics(self) -> None:
        metrics = pandas.DataFrame([ins.metrics for ins in self.instances.values()])
        metrics = metrics.round(3)
        if self.output:
            metrics.to_csv(self.output / "metrics.tsv", sep="\t", index=False)
        self.metrics = metrics

    def is_finished(self, instance) -> bool:
        # if hasattr(instance, "source_finished_reading"): # NOTE: tampered here
        #     return instance.source_finished_reading
        return instance.finish_prediction

    def __call__(self, system):
        system.reset()
        proportion_of_wait_tokens = []
        for instance in self.instance_iterator:  # NOTE: `instance_iterator` returns instances (sentences)
            # NOTE: I changed <is_finished>, because for S2TT it returned before predictions were finished.
            while not self.is_finished(instance):  # NOTE: instances are sentences
                input_segment = instance.send_source(self.source_segment_size)  # segments are words
                output_segment = system.pushpop(input_segment)  # NOTE: call policy, get content, store
                instance.receive_prediction(output_segment)
                if instance.finish_prediction:
                    # if instance.finish_prediction where set by the reader,
                    # source_finished_reading will be set as well. If it is
                    # set by any of the intermediate components, then we didn't
                    # end yet. We are going to clear the state and continue
                    # processing the rest of the input.
                    try:
                        if system._japanese_target:
                            items_in_prediction = len("".join(instance.prediction_list))
                        else:
                            items_in_prediction = len(instance.prediction_list)

                        cprint(f'WAIT TOKENS IN SENTENCE: {system.WAIT_TOKENS_IN_SENTENCE}')
                        cprint(f'ITEMS IN PREDICTION: {items_in_prediction}')
                        prop = system.WAIT_TOKENS_IN_SENTENCE / items_in_prediction
                    except Exception as e:
                        cprint(e, color='red')
                        prop = 0.0
                    cprint(f'PROPORTION OF WAIT TOKENS IN SENTENCE: {prop}', 'grey', 'on_cyan', attrs=['bold'])
                    proportion_of_wait_tokens.append(prop)
                    system.reset()

            if not self.score_only:
                self.write_log(instance)

        results = self.dump_results()  # results, metrics are computed once all the instances (sentences) are processed
        self.dump_metrics()

        if not system._japanese_target:
            predictions = "\n".join([i.prediction for i in self.instances.values()])
            references = fix_spaces_around_punctuation("\n".join([i.reference[:-1] for i in self.instances.values()]))
        else:
            predictions = ("\n".join([i.prediction for i in self.instances.values()])).replace(" ", "")
            references = ("\n".join([i.reference[:-1] for i in self.instances.values()])).replace(" ", "")
        cprint(predictions, color='yellow')
        cprint(references, color='magenta')

        cprint(f'proportion_of_wait_tokens : {proportion_of_wait_tokens}', color='magenta', attrs=['bold'])
        return results, predictions, references, np.mean(proportion_of_wait_tokens)

    @classmethod
    def from_args(cls, args):
        if not args.score_only:
            dataloader = build_dataloader(args)
        else:
            dataloader = None

        latency_scorers = {}
        use_ref_len = not args.no_use_ref_len
        for name in args.latency_metrics:
            latency_scorers[name] = get_scorer_class("latency", name).from_args(args)
            if args.computation_aware:
                latency_scorers[name + "_CA"] = get_scorer_class("latency", name)(computation_aware=True,
                                                                                  use_ref_len=use_ref_len)

        quality_scorers = {}
        for name in args.quality_metrics:
            quality_scorers[name] = get_scorer_class("quality", name).from_args(args)

        return cls(dataloader, quality_scorers, latency_scorers, args)