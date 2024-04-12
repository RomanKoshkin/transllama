# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from re import I
from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from termcolor import cprint
from nltk import word_tokenize


@entrypoint
class StoneFishAgent(TextToTextAgent):

    tgt = word_tokenize(
        "И сегодня вы управляете крупнейшим бизнес-дивизионом крупнейшей компании в мире, которая занимается продажей продуктов питания."
    )
    tid = 0
    finieshed = False

    def policy(self):
        prediction = self.tgt[self.tid]
        # prompt = make_prompt(self.states.source, self.states.target, etc...)

        # prediction = model.generate(prompt)
        # while not prediction.last_word:
        # if prediction.last_token == 2:
        # finished = True
        # break

        #   return WriteAction(prediction.last_word, finished=finished)

        finished = prediction == "."

        # condition for not emitting, but just reading: <WAIT_TOKEN>
        if self.states.source[-1] == "the":
            return ReadAction()

        if not self.states.source_finished:
            pass
        else:
            cprint(self.states.source, color='blue')

        self.tid = self.tid + 1 if not finished else 0
        return WriteAction(prediction, finished=finished)


# @entrypoint
# class DummyWaitkTextAgent(TextToTextAgent):
#     waitk = 3
#     vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

#     def policy(self):
#         lagging = len(self.states.source) - len(self.states.target)
#         print(f'lagging: {lagging}')
#         cprint(f"{self.states.source} : {self.states.target}", color='red')

#         # here we determine whether we Read or Write
#         if lagging >= self.waitk or self.states.source_finished:
#             prediction = random.choice(self.vocab)

#             return WriteAction(prediction, finished=(lagging <= 1))
#         else:
#             return ReadAction()
