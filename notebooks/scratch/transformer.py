
import cProfile
import pstats
import time
from datetime import date, datetime
import torch as t
from pathlib import Path
import tokenizers as hf
from infra import tokens
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset
from infra.inference import generate_text
from infra.logger import DataLogger
from infra.run import Run, RunConfig
from infra.models import Transformer
from datasets import load_dataset

vocab_size = 1000
context_size = 2048
# %%

# %%
# hf_tokenizer = hf.Tokenizer(hf.models.BPE(unk_token="[UNK]"))
# hf_trainer = hf.trainers.BpeTrainer(
#     special_tokens=["[UNK]", "[END]"], vocab_size=vocab_size
# )
# hf_tokenizer.pre_tokenizer = hf.pre_tokenizers.Whitespace()

# hf_tokenizer.train(files=["/home/dan/Downloads/TinyStories-valid.txt"], trainer=hf_trainer)
# hf_tokenizer.enable_padding()

# %%

stories = load_dataset("roneneldan/TinyStories", split="validation").with_format("torch")
tokenizer = tokens.Tokenizer.train(stories['text'][:1000], vocab_size)

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset['text']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

def collate_fn(data):
    tokens = t.tensor(tokenizer.encode_batch(data))
    # TODO: is this right? labels should be offset by 1
    labels = tokens.view(-1)

    return tokens, labels

stories = CustomDataset(stories)
model = Transformer(
    d_model=256,
    nhead=4,
    num_decoder_layers=2,
    dim_feedforward=1024,
    vocab_size=1000,
    context_size=3000,
)

train_loader = DataLoader(stories, batch_size=32, shuffle=True, collate_fn=collate_fn)
run_dir = Path("./notebooks/scratch/transformer_run")
config = Runonfig(val_interval=0, epochs=10)

run = Run(config, run_dir)
run.start(train_loader, model=model)
# TODO: handle texts longer than seq_len (truncate?)
# %%
ids = []
for _ in range(100):
    logits = model(t.tensor([ids]))
    new_id = t.nn.functional.softmax(logits[0][0]).argmax()
    ids.append(new_id)
    if new_id == tokenizer.ids[tokenizer.vocab.end_of_string]:
        break

tokenizer.decode(ids)

# %%
# tokens = tokenizer.encode_batch(stories[:1]["text"])


# tokens = t.tensor([t.ids for t in tokens[:1]]).to(t.long)
# x = tokens[0][0].unsqueeze(0).unsqueeze(0)
# x
# y = model(x)
# y

# foo = t.nn.functional.softmax(y[0][0]).argmax()
# tokenizer.decode([foo])
# %%

# import timeit
# story = stories[0]['text']

# def tok_func():
#     tokenizer.encode(story)

# def hf_func():
#     hf_tokenizer.encode(story)

# # tok_func()
# # tokens = tokenizer.encode(stories[1]['text'])
# # tokens = tokenizer.encode_batch(stories[:8]['text'])
# tok_time = timeit.timeit(tok_func, number=100)
# # hf_time = timeit.timeit(hf_func, number=100)
# print(f"Execution time: {tok_time} seconds")
# # print(f"HF Execution time: {hf_time} seconds")
# # Run the profiler
# # cProfile.run('tokenizer.encode_batch(stories["text"][:32])', sort='cumtime')
# # cProfile.run('tok_func()', sort='cumtime')


# # Read and sort the statistics
# # with open('profile_stats', 'rb') as stats_file:
# #     p = pstats.Stats(stats_file)
# #     p.sort_stats('cumulative').print_stats(10)
# # profiler = cProfile.Profile()
# # profiler.enable()
# # tok_func()
# # profiler.disable()

# # # Save profiling results to a file
# # with open('profile_results.txt', 'w') as f:
# #     stats = pstats.Stats(profiler, stream=f)
# #     stats.sort_stats('cumulative')
# #     stats.print_stats()

# # Generate flame graph

# # %%
# from infra import tokens
# tokens.Tokenizer.__mro__
