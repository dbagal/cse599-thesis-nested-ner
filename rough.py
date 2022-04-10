import torch

y =[
    [1,1,0,1,0,1],
    [1,1,1,1,1,1],
    [0,0,1,0,1,1],
    [1,0,1,0,1,1]
]
y = [y,y,y,y]

ypred = [
    [1,1,0,0,0,0],
    [0,0,1,1,1,1],
    [1,1,0,0,1,0],
    [0,0,0,0,1,1]
]
ypred = [ypred, ypred, ypred, ypred]

y = torch.FloatTensor(y)
ypred = torch.FloatTensor(ypred)
tp = torch.mul(y == ypred, ypred==1.0).sum(dim=0)
tn = torch.mul(y == ypred, ypred==0.0).sum(dim=0)
fp = torch.mul(y!=ypred, ypred==1.0).sum(dim=0)
fn = torch.mul(y!=ypred, ypred==0.0).sum(dim=0)

print(tn)