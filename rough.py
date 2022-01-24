import torch
from torch.nn import functional as f

a = [
    [
        [0,0,1,2,3,4,5,0,0]
    ]
]
a = torch.tensor(a)
print(a.shape) # 1,1,9

b = f.unfold(a.view(1,*a.shape), kernel_size=(5,1))
print(b)