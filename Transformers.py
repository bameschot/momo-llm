import torch
import torch.nn as nn

from GPTModelConfig import *
from SelfAttention import MultiHeadAttention
from FeedForward import FeedForward
from LayerNormalization import LayerNormalization

class GPTTransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feedForward = FeedForward(config)
        self.normalizationLayer1 = LayerNormalization(config[EMBEDDING_DIMENSION])
        self.normalizationLayer2 = LayerNormalization(config[EMBEDDING_DIMENSION])
        self.dropoutShortcut = nn.Dropout(config[DROPOUT_SHORTCUT_RATE])

    def forward(self,x):
        shortcut = x
        x = self.normalizationLayer1(x)
        x = self.attention(x)
        x = self.dropoutShortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.normalizationLayer2(x)
        x = self.feedForward(x)
        x = self.dropoutShortcut(x)
        x = x + shortcut

        return x
    
# torch.manual_seed(123)
# x = torch.rand(2,4,768)
# block = GPTTransformerBlock(GPT_CONFIG_124M)
# output = block(x)
# print(f"output: {output}")
# print(f"input shape {x.shape}")
# print(f"output shape {output.shape}")

