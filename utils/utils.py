import sys

sys.path.append('../')
from transformers import StoppingCriteriaList, StoppingCriteria
import torch
import torch.nn as nn
from typing import Tuple
from tqdm import trange

from torch.utils.data import dataset
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer

import yaml, wandb, os
from einops import rearrange
import numpy as np
import math

from torch.utils.data import dataset
import time


def tonp(tensor):
    return tensor.detach().cpu().numpy()


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class AttributeDict(dict):
    """ convenience class. To get and set properties both as if it were a dict or obj """
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_config(pathAndNameOfConfig):
    yaml_txt = open(pathAndNameOfConfig).read()
    parms_dict = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    parms_obj = AttributeDict()
    for k, v in parms_dict.items():
        parms_obj[k] = v
    return parms_obj


class EndOfSentenceCriteria(StoppingCriteria):

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[:, -1] == 50256:
            return True  # The end-of-sentence token in GPT-2 is 50256, Llama 13, vicuna 835
        else:
            return False


def data_process(raw_text_iter: dataset.IterableDataset, tokenizer, vocab) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source: torch.Tensor, i: int, bptt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return data.T, target.T.reshape(-1)


class Timer:

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f'{self.interval:.4f} s.')
        if type is KeyboardInterrupt:
            print("KeyboardInterrupt caught. Cleaning up...")
            return True  # This will suppress the exception


class Trainer:

    def __init__(self, model, config, use_wandb=True) -> None:
        self.use_wandb = use_wandb
        self.config = config
        self.model = model
        train_iter = WikiText2(split='train')
        tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        # ``train_iter`` was "consumed" by the process of building the vocab,
        # so we have to create it again
        train_iter, val_iter, test_iter = WikiText2()

        train_data = data_process(train_iter, tokenizer, self.vocab)
        val_data = data_process(val_iter, tokenizer, self.vocab)
        test_data = data_process(test_iter, tokenizer, self.vocab)

        # make batches
        self.train_data = batchify(
            train_data,
            self.config.batch_size,
        ).to(self.config.DEVICE)  # shape ``[seq_len, batch_size]``
        self.val_data = batchify(
            val_data,
            self.config.eval_batch_size,
        ).to(self.config.DEVICE)
        self.test_data = batchify(
            test_data,
            self.config.eval_batch_size,
        ).to(self.config.DEVICE)

        lr = 5.0  # learning rate
        self.optimizer = torch.optim.AdamW(params=self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # if you just want one run (doesn't work with sweeps)
        if self.use_wandb:
            wandb.init(
                project="SimpliFormer",
                entity="nightdude",
                config=dict(self.config),
                save_code=False,
            )

    def _train_epoch(self):
        self.model.train()
        losses = []

        for b, i in enumerate(trange(0, self.train_data.size(0) - 1, self.config.bptt, bar_format=bar_format)):

            data, targets = get_batch(self.train_data, i, self.config.bptt)
            x = data.to(self.config.DEVICE)  # (B, bptt)
            self.optimizer.zero_grad()

            out = self.model(x)

            loss = self.criterion(
                rearrange(out, 'b t v -> (b t) v'),
                targets,
            )
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        mean_loss = np.mean(losses)
        print(f"train_loss: {mean_loss:.2f}")
        return mean_loss

    def evaluate(self):
        self.model.eval()
        losses = []

        for b, i in enumerate(trange(0, self.val_data.size(0) - 1, self.config.bptt, bar_format=bar_format)):

            with torch.no_grad():
                data, targets = get_batch(self.val_data, i, self.config.bptt)
                x = data.to(self.config.DEVICE)  # (B, bptt)

                out = self.model(x)

                loss = self.criterion(
                    rearrange(out, 'b t v -> (b t) v'),
                    targets,
                )
                losses.append(loss.item())
        mean_loss = np.mean(losses)
        print(f"val_loss: {mean_loss:.2f}")
        return mean_loss

    def train(self, EPOCHS):

        val_loss = np.nan
        for ep in trange(EPOCHS):
            train_loss = self._train_epoch()
            if ep % 10:
                val_loss = self.evaluate()

            try:
                lr = self.scheduler.get_last_lr()[-1]
            except:
                lr = 0.0

            if self.use_wandb:
                wandb.log(
                    dict(
                        train_loss=train_loss,
                        train_ppl=math.exp(train_loss),
                        val_loss=val_loss,
                        val_ppl=math.exp(val_loss),
                        lr=lr,
                    ))
            # self.scheduler.step()
