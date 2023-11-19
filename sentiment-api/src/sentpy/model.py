import torch 

class Model(torch.nn.Module):
    def __init__(self, vocab_size: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size, 20_000)
        self.linear2 = torch.nn.Linear(20_000, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.linear2(x)