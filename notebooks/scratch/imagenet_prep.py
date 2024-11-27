import os
import shutil
from tqdm import tqdm
from pathlib import Path

train = Path(".data/imagenet-1k/train")
val = Path(".data/imagenet-1k/val")
train_out = Path(".data/imagenet-1k-parsed/train")
val_out = Path(".data/imagenet-1k-parsed/val")

step = 0
with tqdm(total=1285044) as prog:
    for file in train.glob("*"):
        file = Path(file)
        parts = file.stem.split('_')
        category = parts[-1]
        id = parts[-2]
        Path(f"{train_out}/{category}").mkdir(exist_ok=True, parents=True)
        shutil.copy(file, Path(f"{train_out}/{category}/{id}.jpg"))
        step +=1
        if step % 100 == 0:
            prog.update(100)
