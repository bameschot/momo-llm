import torch
import torch.nn as nn

class SelfAttention (nn.Module):
    def __init__(self,dimensionIn,dimensionOut,keyValueBias=False):
        super().__init__()
        self.wQuery = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wKey = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wValue = nn.Linear(dimensionIn,dimensionOut,keyValueBias)

    def forward(self, input:torch.Tensor):
        queries = self.wQuery(input)
        keys = self.wKey(input)
        values = self.wValue(input)

        attentionScores = queries @ keys.T
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1] ** 0.5,dim=-1)
        print(attentionWeights)
        contextVectors = attentionWeights @ values
        return contextVectors

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(123)
sav1 = SelfAttention(3,2)
contextVectors = sav1.forward(inputs)
print("tensor([0.3061, 0.8210])")
print(contextVectors)        