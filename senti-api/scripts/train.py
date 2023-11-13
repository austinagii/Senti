from collections.abc import Generator
from os import path 

import datasets
import torch
from torch.optim import SGD

CHECKPOINTS_DIR = path.join(path.dirname(path.abspath(__file__)), '..', 'checkpoints')

class Tokenizer:
    def __init__(self, dataset: datasets.Dataset):
        self.vocab = set(['UNK']) | set((word for document in dataset for word in document['text'].split()))
        self.vocab_size = len(self.vocab)
        self.id_by_word = {word: id for id, word in enumerate(self.vocab)}

    def tokenize(self, document: str) -> list[int]:
        return [self.id_by_word[(word if word in self.vocab else 'UNK')] for word in document.split()]
    

class Model(torch.nn.Module):
    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size, 20_000)
        self.linear2 = torch.nn.Linear(20_000, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.linear2(x)


def to_batches(
    dataset: datasets.Dataset) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch['text']) < batch_size:
            continue
        labels = torch.tensor(batch['label'], dtype=torch.long)
        bows = torch.zeros((batch_size, tokenizer.vocab_size))
        for j, document in enumerate(batch['text']):
            bows[j, tokenizer.tokenize(document)] = 1
        yield bows, labels


if __name__ == '__main__':
    batch_size = 32
    n_classes = 6
    stopping_criterion = 1e-1

    dataset = datasets.load_dataset("emotion")
    tokenizer = Tokenizer(dataset['train'])
    model = Model(len(tokenizer.vocab), n_classes)
    optimizer = SGD(model.parameters(), lr=0.01)

    print("Model initalized, starting training...")
    min_loss = float('inf')
    epoch = 0
    iterations_without_improvement = 0
    while iterations_without_improvement < 3:
        epoch += 1
        total_loss = 0
        for batch_no, batch in enumerate(to_batches(dataset['train']), start=1):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch[0]), batch[1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / batch_no

        with torch.no_grad():
            labels = torch.tensor(dataset['validation']['label'], dtype=torch.long)
            bows = torch.zeros((len(dataset['validation']), tokenizer.vocab_size))
            for i, document in enumerate(dataset['validation']['text']):
                bows[i, tokenizer.tokenize(document)] = 1
            val_loss = torch.nn.functional.cross_entropy(model(bows), labels)

        if val_loss < min_loss - stopping_criterion:
            min_loss = val_loss
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        print(f"Epoch #{epoch} Avg train loss: {avg_loss:.3f} val loss: {val_loss:.3f} iterations without improvement: {iterations_without_improvement}")
        break 


    traced_model = torch.jit.trace(model, torch.randn((1, tokenizer.vocab_size)))
    traced_model.save(path.join(CHECKPOINTS_DIR, 'model.pt'))
    #torch.save(model.state_dict(), 'sentiment_model.pth')
