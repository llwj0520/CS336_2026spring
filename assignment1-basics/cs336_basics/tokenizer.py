from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import regex as re


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
    if not special_tokens:
        return [(text, False)]

    # 长的 special token 要先匹配，避免 "<|endoftext|><|endoftext|>" 被拆成两个短 token。
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(token) for token in sorted_special_tokens))

    pieces = []
    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            pieces.append((text[start : match.start()], False))
        pieces.append((match.group(0), True))
        start = match.end()

    if start < len(text):
        pieces.append((text[start:], False))

    return pieces


def _word_to_byte_tuple(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in word.encode("utf-8"))


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word

    merged_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged_word.append(word[i] + word[i + 1])
            i += 2
        else:
            merged_word.append(word[i])
            i += 1

    return tuple(merged_word)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    for byte in range(256):
        vocab[len(vocab)] = bytes([byte])

    if vocab_size < len(vocab):
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")

    text = Path(input_path).read_text(encoding="utf-8")

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for piece, is_special in _split_on_special_tokens(text, special_tokens):
        if is_special:
            continue
        for match in PAT.finditer(piece):
            word = _word_to_byte_tuple(match.group(0))
            if word:
                word_counts[word] += 1

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for word, count in word_counts.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count

        if not pair_counts:
            break

        # 频率最高的 pair 被 merge；频率相同，按 bytes 的字典序打破平局。
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        new_word_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_counts.items():
            new_word_counts[_merge_word(word, best_pair)] += count
        word_counts = new_word_counts

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_token_to_id = {
            token: self.byte_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.byte_to_id
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw_vocab.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip("\n").split("\t")
                merges.append((bytes.fromhex(left), bytes.fromhex(right)))

        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike) -> None:
        raw_vocab = {token_id: token_bytes.hex() for token_id, token_bytes in self.vocab.items()}
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_vocab, f)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{left.hex()}\t{right.hex()}\n")

    def _encode_word(self, word: str) -> list[int]:
        tokens = _word_to_byte_tuple(word)

        while len(tokens) >= 2:
            best_rank = None
            best_pair = None

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = _merge_word(tokens, best_pair)

        return [self.byte_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids = []

        for piece, is_special in _split_on_special_tokens(text, self.special_tokens):
            if is_special:
                ids.append(self.special_token_to_id[piece])
                continue

            for match in PAT.finditer(piece):
                ids.extend(self._encode_word(match.group(0)))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")


def encode_text_file_to_npy(
    tokenizer: Tokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    dtype=np.uint16,
) -> None:
    token_ids = np.fromiter(tokenizer.encode_iterable(Path(input_path).open(encoding="utf-8")), dtype=dtype)
    np.save(output_path, token_ids)
