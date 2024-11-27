import torch

def generate_text(model, tokenizer, max_len):
    ids = []
    for _ in range(max_len):
        logits = model(ids)
        new_id = torch.nn.functional.softmax(logits[0][0]).argmax()
        ids.append(new_id)
        if new_id == tokenizer.ids[tokenizer.vocab.end_of_string]:
            break

    return tokenizer.decode(ids)
