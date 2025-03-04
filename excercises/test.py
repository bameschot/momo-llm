import time
import torch

import torch.nn as nn



class CNN(nn.Module):
    def __init__(self ,device,NumBins=32):
        self.InNodes=int(NumBins)*2
        self.MediumNode=self.InNodes*2
        super(CNN, self).__init__()
        self.Lin1 = nn.Linear(self.InNodes , self.MediumNode).to(device=device)
        self.Lin2 = nn.Linear(self.MediumNode,  self.MediumNode).to(device=device)
        self.Lin5 = nn.Linear(self.MediumNode, 2).to(device=device)
    def forward(self, input):
        gelu = nn.ReLU()
        Zoutput = self.Lin1(input)
        Zoutput = gelu(Zoutput)
        Zoutput = self.Lin2(Zoutput)
        Zoutput = gelu(Zoutput)
        Zoutput = self.Lin5(Zoutput)
        return Zoutput
    

x = torch.randn(1000000, 64)

device = torch.device('cpu')
model = CNN(device)

cpu_times = []

for epoch in range(100):
    t0 = time.perf_counter()
    output = model(x)
    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = torch.device('mps')
model = CNN(device=device)
model = model.to(device)
x = x.to(device)
torch.mps.synchronize()

gpu_times = []
for epoch in range(100):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    output = model(x)
    torch.mps.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print('CPU {}, GPU {}'.format(
    torch.tensor(cpu_times).mean(),
    torch.tensor(gpu_times).mean()))
