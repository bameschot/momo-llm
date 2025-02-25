import torch
import torch.nn as nn

import GPTModelConfig

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(
                torch.tensor(2 / torch.pi) * (x + 0.044715 * torch.pow(x,3))
            ))
        )

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config[GPTModelConfig.EMBEDDING_DIMENSION],4 * config[GPTModelConfig.EMBEDDING_DIMENSION]),
            GELU(),
            nn.Linear(4 * config[GPTModelConfig.EMBEDDING_DIMENSION], config[GPTModelConfig.EMBEDDING_DIMENSION])
        )

    def forward(self,x):
        return self.layers(x)