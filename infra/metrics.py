def multiclass_accuracy(logits, labels):
    assert len(logits) == len(labels)
    preds = logits.argmax(dim=1)
    correct = preds == labels
    accuracy = correct.sum().item() / len(logits)
    return accuracy
