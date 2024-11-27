import torch as t
import infra.tokens as tok
import tokenizers as hf
from datasets import load_dataset

vocab_size = 5000
context_size = 2048
stories = load_dataset("roneneldan/TinyStories", split="validation").with_format("torch")
# tokenizer = hf.Tokenizer(hf.models.BPE(unk_token="[UNK]"))
# trainer = hf.trainers.BpeTrainer(
#     special_tokens=["[UNK]", "[END]"], vocab_size=vocab_size
# )
# tokenizer.pre_tokenizer = hf.pre_tokenizers.Whitespace()

# tokenizer.train(files=["/home/dan/Downloads/TinyStories-valid.txt"], trainer=trainer)
# tokenizer.enable_padding()
# tokens = tokenizer.encode_batch(stories[:10]["text"])
# tokens = t.Tensor([t.ids for t in tokens]).to(t.long)
# tokens
print(len(stories))
# %%


story = stories[0]['text']
tokenizer = tok.Tokenizer.train(stories['text'][:10000], 1000)
tokens = tokenizer.encode(story)
print(tokens)
print(tokenizer.decode(tokens))


# %%
hf_tokenizer = hf.Tokenizer(hf.models.BPE())
trainer = hf.trainers.BpeTrainer(vocab_size=1000)
hf_tokenizer.pre_tokenizer = hf.pre_tokenizers.Whitespace()
hf_tokenizer.train_from_iterator([stories['text'][:10000]], trainer)
hf_vocab = hf_tokenizer.get_vocab()
print(len(vocab.tokens))
print(len(hf_vocab.keys()))
print(sorted(list(vocab.tokens)))
print(sorted(list(hf_vocab.keys())))
# %%
# tok.multi_partition("foo.p\no,", '\n.,')
parts = tok.pre_tokenize(story)
# parts = hf_tokenizer.pre_tokenizer.pre_tokenize_str(story)
print(parts)
# string = ""
# for part, (start, end) in parts:
#     if start > len(string):
#         string += ' '
#     string += part
print(story)

# print(string)
# %%


# print(list(zip(sorted(vocab.tokens), sorted(list(hf_vocab.keys())))))
print(len(set(vocab.tokens) & set(list(hf_vocab.keys()))))
# %%
tokens = tokenizer.encode(story)
hf_tokens = hf_tokenizer.encode(story)

print(len(tokens))
print(len(hf_tokens.ids))
# %%
batch = tokenizer.encode_batch(stories['text'][:10])
print(tokenizer.decode(batch[0]))
print([len(toks) for toks in batch])
print(t.tensor(batch))
print(len(batch))
# for story in stories['text'][:10]:
#     print(story[:40])
