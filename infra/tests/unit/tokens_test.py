from infra.tokens import Tokenizer
import random

test_corpus = ["hug"] * 10 + ["pug"] * 5 + ["pun"] * 12 + ["bun"] * 4 + ["hugs"] * 5
random.shuffle(test_corpus)
test_corpus = ", ".join(test_corpus)

def test_train():
    tokenizer = Tokenizer.train(test_corpus, 1000)
    expected_vocab = ['[PAD]', '[UNK]', '[EOS]', 'p', 'u', 'n', ',', ' ', 'g', 'h', 's', 'b', 'ug', 'un', 'hug', 'pun', 'pug', 'hugs', 'bun']

    assert sorted(tokenizer.tokens) == sorted(expected_vocab)
    assert ('h', 'ug') in tokenizer.merges.keys()

def test_encode_decode():
    tokenizer = Tokenizer.train(test_corpus, 1000)
    ids = tokenizer.encode('hug un')

    assert len(ids) == 4
    assert isinstance(ids[0], int)
    assert tokenizer.vocab.ids_to_tokens(ids) == ['hug', ' ', 'un', '[EOS]']
    assert tokenizer.decode(ids) == 'hug un[EOS]'
