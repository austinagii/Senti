from os import path

import torch 

from train import Model

CHECKPOINTS_DIR = path.join(path.dirname(path.abspath(__file__)), '..', 'checkpoints')

model = Model(15213, 6)
model.load_state_dict(torch.load(path.join(CHECKPOINTS_DIR, 'sentiment_model.pth')))
model.eval()

x = torch.randn((1, 15213))
traced_script_module = torch.jit.trace(model, x)

traced_script_module.save(path.join(CHECKPOINTS_DIR, 'sentiment_model.pt'))