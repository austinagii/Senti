import json 
from os import path 

import datasets
import torch
from torch.optim import SGD

from sentpy.preprocessing import to_batches, Tokenizer
from sentpy.model import Model


if __name__ == '__main__':
    batch_size = 32
    n_classes = 6
    stopping_criterion = 1e-3
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # dev = torch.device("cpu")
    dataset = datasets.load_dataset("emotion")
    tokenizer = Tokenizer(dataset['train'])
    model = Model(len(tokenizer.vocab), n_classes).to(dev)
    optimizer = SGD(model.parameters(), lr=0.01)

    print(f"Model initalized, starting training on '{dev}'...")
    min_loss = float('inf')
    epoch = 0
    iterations_without_improvement = 0
    while iterations_without_improvement < 3:
        epoch += 1
        total_loss = 0
        train_batches = to_batches(batch_size, dataset['train'], tokenizer)
        for batch_no, batch in enumerate(train_batches, start=1):
            optimizer.zero_grad()
            X, y = batch[0].to(dev), batch[1].to(dev)
            loss = torch.nn.functional.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / batch_no

        with torch.no_grad():
            labels = torch.tensor(dataset['validation']['label'], dtype=torch.long).to(dev)
            bows = torch.zeros((len(dataset['validation']), tokenizer.vocab_size)).to(dev)
            for i, document in enumerate(dataset['validation']['text']):
                bows[i, tokenizer.tokenize(document)] = 1
            val_loss = torch.nn.functional.cross_entropy(model(bows), labels)

        if val_loss < min_loss - stopping_criterion:
            min_loss = val_loss
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        print(f"Epoch #{epoch} Avg train loss: {avg_loss:.3f} val loss: {val_loss:.3f} iterations without improvement: {iterations_without_improvement}")

    output_dir = path.join(path.dirname(path.abspath(__file__)), '..', 'artifacts')
    # Save the tokenizer state.
    with open(path.join(output_dir, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer.id_by_word, f)

    # Save the model.
    with torch.no_grad():
        r = torch.randn((1, tokenizer.vocab_size)).to(dev)
        traced_model = torch.jit.trace(model, r)
        traced_model.save(path.join(output_dir, 'model.pt'))
