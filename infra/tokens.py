from collections import Counter, abc
import time
from datetime import datetime
import re

from tqdm import tqdm


class VocabFullError(ValueError):
    def __init__(self, max_size):
        self.max_size = max_size
        super().__init__(f"Cannot add token. Vocabulary is already at maximum size of {max_size}")


class Vocab:
    padding = "[PAD]"
    unknown = "[UNK]"
    end_of_string = "[EOS]"

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.tokens = [self.padding, self.unknown, self.end_of_string]
        # TODO: this should be a trie for mem efficiency
        self.ids = {self.padding: 0, self.unknown: 1, self.end_of_string: 2}

    # TODO: handle index errors
    def id_to_token(self, id: int):
        return self.tokens[id]

    def ids_to_tokens(self, ids: abc.Iterable[int]):
        return [self.id_to_token(id) for id in ids]

    def token_to_id(self, token: str):
        try:
            return self.ids[token]
        except KeyError:
            return self.ids[self.unknown]

    def add_token(self, token: str) -> int:
        if len(self) >= self.max_size:
            raise VocabFullError(self.max_size)
        try:
            return self.ids[token]
        except KeyError:
            id = len(self)
            self.tokens.append(token)
            self.ids[token] = id
            return id

    def add_tokens(self, tokens: abc.Iterable):
        ids = []
        for token in tokens:
            ids.append(self.add_token(token))

        return ids

    def __len__(self):
        return len(self.tokens)


def normalize_text(corpus: str) -> str:
    # Not much needed for tiny stories - will need to at least add unicode normalization later, and/or switch to byte-based
    corpus = corpus.strip()
    return corpus


def pre_tokenize(corpus: str) -> list[str]:
    # Not tracking offsets b/c not really needed for autoregressive task, and can prob recover original text b/c not removing chars
    # TODO: reconsider having a separate token for spaces, and not allowing grouping of punctuation - it makes sequences significantly longer
    splits = re.split(r'([,\.\'"!#$%^&*(){}\[\];:<>/?\\|`~\-_=+\s])', corpus)
    splits = [split for split in splits if len(split) > 0]

    return splits


class Tokenizer:
    # TODO: add save/load from file
    def __init__(self, vocab: Vocab, merges):
        self.vocab = vocab
        self.merges = merges

    @property
    def tokens(self):
        return self.vocab.tokens

    @property
    def ids(self):
        return self.vocab.ids

    # TODO: accept file(s)
    @classmethod
    def train(cls, corpus, vocab_size: int):
        if isinstance(corpus, str):
            corpus = [corpus]

        vocab = Vocab(vocab_size)
        merges = {}
        corpus = [normalize_text(item) for item in corpus]
        words = []
        for item in corpus:
            words += pre_tokenize(item)

        freqs = Counter()
        for word in words:
            chars = tuple(word)
            freqs[chars] += 1
            vocab.add_tokens(chars)

        with tqdm(total=vocab_size, initial=len(vocab)) as pbar:
            while True:
                pairs = {}
                for word, freq in freqs.items():
                    word_pairs = list(zip(word, word[1:]))
                    for idx, pair in enumerate(word_pairs):
                        pair_info = pairs.get(pair, {"count": 0, "words": []})
                        pair_info["count"] += freq
                        pair_info["words"].append((idx, word))
                        pairs[pair] = pair_info

                if len(pairs) == 0:
                    break
                top_pair, top_pair_info = sorted(
                    pairs.items(), key=lambda x: x[1]["count"], reverse=True
                )[0]
                new_token = "".join(top_pair)
                try:
                    vocab.add_token(new_token)
                except VocabFullError:
                    break

                merges[top_pair] = len(merges)
                for idx, word in top_pair_info["words"]:
                    freq = freqs[word]
                    del freqs[word]
                    new_word = list(word[:idx])
                    new_word.append(new_token)
                    new_word += word[idx + 2 :]
                    freqs[tuple(new_word)] = freq
                pbar.update()

            return cls(vocab, merges)

    def bpe_word(self, word):
        tokens = list(word)

        def possible_merge_sort(x):
            return x[0]

        while len(tokens) > 1:
            possible_merges = []
            for i in range(len(tokens) - 1):
                try:
                    pair = (tokens[i], tokens[i + 1])
                    possible_merges.append((self.merges[pair], pair))
                except KeyError:
                    pass
            if len(possible_merges) == 0:
                break
            possible_merges.sort(key=possible_merge_sort)
            priority_merge = possible_merges[0][1]
            for idx in range(len(tokens)):
                if tokens[idx] == priority_merge[0] and tokens[idx + 1] == priority_merge[1]:
                    merge_idx = idx
                    break
            tokens = [*tokens[:merge_idx], "".join(priority_merge), *tokens[merge_idx + 2 :]]
        return tokens

    def encode(self, text, add_eos=True):
        text = normalize_text(text)
        words = pre_tokenize(text)
        words = [self.bpe_word(word) for word in words]
        ids = [self.vocab.token_to_id(token) for word_tokens in words for token in word_tokens]
        if add_eos:
            ids.append(self.vocab.ids[self.vocab.end_of_string])

        return ids

    def encode_batch(self, batch):
        encodings = [self.encode(text) for text in batch]

        max_len = max([len(tokens) for tokens in encodings])
        padding_id = self.vocab.ids[self.vocab.padding]
        padded_encodings = []
        for encoding in encodings:
            if len(encoding) == max_len:
                padded_encodings.append(encoding)
            else:
                padded = encoding + [padding_id] * (max_len - len(encoding))
                padded_encodings.append(padded)
        return padded_encodings

    def decode(self, ids):
        tokens = [self.vocab.id_to_token(id) for id in ids]
        return "".join(tokens)
