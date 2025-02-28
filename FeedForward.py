import torch
import torch.nn as nn

from GPTModelConfig import *

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))
            )
        )

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            #expand
            nn.Linear(config[EMBEDDING_DIMENSION],4 * config[EMBEDDING_DIMENSION]),
            #activate
            GELU(),
            #contract
            nn.Linear(4 * config[EMBEDDING_DIMENSION], config[EMBEDDING_DIMENSION])
        )

    def forward(self,x):
        return self.layers(x)
    