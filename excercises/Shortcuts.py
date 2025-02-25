import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self,layerSizes,useShortcut = False):
        super().__init__()
        self.useShortcut = useShortcut
        self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(layerSizes[0],layerSizes[1]),nn.GELU()),
                nn.Sequential(nn.Linear(layerSizes[1],layerSizes[2]),nn.GELU()),
                nn.Sequential(nn.Linear(layerSizes[2],layerSizes[3]),nn.GELU()),
                nn.Sequential(nn.Linear(layerSizes[3],layerSizes[4]),nn.GELU()),
                nn.Sequential(nn.Linear(layerSizes[4],layerSizes[5]),nn.GELU())
            ]
        )

    def forward(self,x):
        for layer in self.layers:
            layerOut = layer(x)
            if self.useShortcut and x.shape == layerOut.shape:
                x = x + layerOut
            else: 
                x = layerOut
        return x
    
torch.manual_seed(123)
layerSize = [3,3,3,3,3,1]
sampleInput = torch.tensor([[1.,0.,-1.]])

modelNoShortcut = ExampleDeepNeuralNetwork(layerSizes=layerSize,useShortcut=True)

def printGradients(model,x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output,target)
    loss.backward()

    for name, param in model.named_parameters():
        if('weight' in name):
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

printGradients(modelNoShortcut,sampleInput)