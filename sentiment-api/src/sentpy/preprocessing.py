from collections.abc import Generator

import datasets
import torch

class Tokenizer:
    def __init__(self, dataset: datasets.Dataset):
        self.vocab = set(['UNK']) | set((word for document in dataset for word in document['text'].split()))
        self.vocab_size = len(self.vocab)
        self.id_by_word = {word: id for id, word in enumerate(self.vocab)}

    def tokenize(self, document: str) -> list[int]:
        return [self.id_by_word[(word if word in self.vocab else 'UNK')] for word in document.split()]
    

def to_batches(
    batch_size: int,
    dataset: datasets.Dataset,
    tokenizer: Tokenizer) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if len(batch['text']) < batch_size:
            continue
        labels = torch.tensor(batch['label'], dtype=torch.long)
        bows = torch.zeros((batch_size, tokenizer.vocab_size))
        for j, document in enumerate(batch['text']):
            bows[j, tokenizer.tokenize(document)] = 1
        yield bows, labels
