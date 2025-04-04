import torch
vec = torch.rand(size=(2, 3, 79, 50))
w = torch.nn.Linear(in_features=50, out_features=25)
result = w(vec)
print(w.weight.shape)
print(result)
print(result.shape)
# when you pass a tensor through a torch.nn.Linear layer of shape (out_features=Y,in_features=X)
# you really do the operation vec @ 
# (A, B, C, X)