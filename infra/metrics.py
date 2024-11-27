import torch


def multiclass_accuracy(logs):
    logits = torch.cat(logs["logits"])
    labels = torch.cat(logs["labels"])
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    mean = correct.mean().item()
    std = correct.std().item()
    return {"accuracy_mean": mean, "accuracy_std": std}


def loss(logs):
    loss = torch.tensor([loss.item() for loss in logs['loss']])
    mean = loss.mean().item()
    std = loss.std().item()
    return {"loss_mean": mean, "loss_std": std}

# TODO:
def perplexity(logs):
    pass
