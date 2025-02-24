import torch
import torch.nn as nn

torch.manual_seed(123)
torch.set_printoptions(sci_mode=False)

batchExample = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6),nn.ReLU())
out = layer(batchExample)
print(out)

mean = out.mean(dim=-1,keepdim=True)
var = out.var(dim=-1,keepdim=True)
print(f"mean {mean}")
print(f"var  {var}")

