import torch
d = torch.nn.Linear(in_features=50, out_features=25)
print(d.weight.shape)