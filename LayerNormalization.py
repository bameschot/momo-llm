import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,embeddingDimension):
        super().__init__()
        self.epsilon=1e-5
        #trainable shift and scale parameters
        self.scale = nn.Parameter(torch.ones(embeddingDimension))
        self.shift = nn.Parameter(torch.zeros(embeddingDimension))
    
    def forward(self,x):
        #calculate the normalized x
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        normalizedX = (x-mean) / torch.sqrt(var+self.epsilon)
        #scale and offset the normalized x with the trainable parameters
        return self.scale * normalizedX + self.shift