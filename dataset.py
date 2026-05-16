"""
dataset.py - Multi30k data pipeline for DA6401 Assignment 3.
"""

import subprocess
import sys
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


def _ensure_spacy_model(model_name: str) -> None:
    try:
        __import__(model_name)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )


def load_spacy_models():
    _ensure_spacy_model("de_core_news_sm")
    _ensure_spacy_model("en_core_web_sm")
    import de_core_news_sm
    import en_core_web_sm

    return de_core_news_sm.load(), en_core_web_sm.load()


def tokenize_de(text: str, spacy_de=None) -> List[str]:
    if spacy_de is None:
        spacy_de, _ = load_spacy_models()
    return [tok.text.lower() for tok in spacy_de(text)]


def tokenize_en(text: str, spacy_en=None) -> List[str]:
    if spacy_en is None:
        _, spacy_en = load_spacy_models()
    return [tok.text.lower() for tok in spacy_en(text)]


class Vocab:
    def __init__(self, stoi: dict, itos: dict) -> None:
        self.stoi = stoi
        self.itos = itos

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, "<unk>")

    def lookup_indices(self, tokens: Sequence[str]) -> List[int]:
        unk_idx = self.stoi["<unk>"]
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def __len__(self) -> int:
        return len(self.stoi)


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_len: int = 100,
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        min_freq: int = 2,
        spacy_de=None,
        spacy_en=None,
    ) -> None:
        super().__init__()
        self.split = split
        self.max_len = max_len
        self.min_freq = min_freq
        self.spacy_de, self.spacy_en = (spacy_de, spacy_en) if spacy_de and spacy_en else load_spacy_models()
        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split,
            trust_remote_code=True,
        )

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if self.src_vocab is None or self.tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        self.data = self.process_data()

    def _extract_pair(self, example: dict) -> Tuple[str, str]:
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        if "translation" in example:
            translation = example["translation"]
            return translation["de"], translation["en"]
        raise KeyError("Could not find German/English sentence pair in example.")

    def build_vocab(self) -> Tuple[Vocab, Vocab]:
        src_counter = Counter()
        tgt_counter = Counter()
        for example in self.dataset:
            src_text, tgt_text = self._extract_pair(example)
            src_counter.update(tokenize_de(src_text, self.spacy_de))
            tgt_counter.update(tokenize_en(tgt_text, self.spacy_en))

        def make_vocab(counter: Counter) -> Vocab:
            stoi = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
            itos = {idx: token for idx, token in enumerate(SPECIAL_TOKENS)}
            next_idx = len(SPECIAL_TOKENS)
            for token, freq in sorted(counter.items()):
                if freq >= self.min_freq and token not in stoi:
                    stoi[token] = next_idx
                    itos[next_idx] = token
                    next_idx += 1
            return Vocab(stoi=stoi, itos=itos)

        return make_vocab(src_counter), make_vocab(tgt_counter)

    def process_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        processed = []
        for example in self.dataset:
            src_text, tgt_text = self._extract_pair(example)
            src_tokens = ["<sos>"] + tokenize_de(src_text, self.spacy_de)[: self.max_len - 2] + ["<eos>"]
            tgt_tokens = ["<sos>"] + tokenize_en(tgt_text, self.spacy_en)[: self.max_len - 2] + ["<eos>"]
            src_ids = torch.tensor(self.src_vocab.lookup_indices(src_tokens), dtype=torch.long)
            tgt_ids = torch.tensor(self.tgt_vocab.lookup_indices(tgt_tokens), dtype=torch.long)
            processed.append((src_ids, tgt_ids))
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 1):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


def get_dataloaders(batch_size: int = 128, max_len: int = 100):
    spacy_de, spacy_en = load_spacy_models()
    train_dataset = Multi30kDataset(
        split="train",
        max_len=max_len,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
    )
    val_dataset = Multi30kDataset(
        split="validation",
        max_len=max_len,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
    )
    test_dataset = Multi30kDataset(
        split="test",
        max_len=max_len,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=1),
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=1),
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=1),
        num_workers=0,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.src_vocab,
        train_dataset.tgt_vocab,
        spacy_de,
    )
