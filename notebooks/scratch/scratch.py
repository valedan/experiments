
import torch
from torch.utils.data import Dataset
from infra.models import Transformer
from infra.inference import generate_text
import datasets as hf_datasets
from infra.tokens import Tokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset["text"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

train_set = TinyStoriesDataset(
    hf_datasets.load_dataset("roneneldan/TinyStories", split="train").with_format("torch")
)[::100]
tokenizer = Tokenizer.train(
    train_set[: 1000], 100
)
# %%

model = Transformer(d_model=128, nhead=2, num_decoder_layers=2, dim_feedforward=512, vocab_size=100, context_size=4000)

state_dict = torch.load('final.pt')

model.load_state_dict(state_dict)
model.eval()

generate_text(model, tokenizer, 100)
