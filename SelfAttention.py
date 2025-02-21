import torch
import torch.nn as nn

class SelfAttentionV2 (nn.Module):
    def __init__(self,dimensionIn,dimensionOut,contextLength,dropoutP,keyValueBias=False):
        super().__init__()
        self.dimensionOut=dimensionOut
        self.wQuery = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wKey = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wValue = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.dropout = torch.nn.Dropout(dropoutP)
        self.register_buffer('mask', torch.triu(torch.ones(contextLength,contextLength),diagonal=1))


    def forward(self, input:torch.Tensor):
        #input data
        print(input.shape)
        batchNr, numberOfTokens, dIn = input.shape

        #get the keys queries and values for each input
        queries = self.wQuery(input)
        keys = self.wKey(input)
        values = self.wValue(input)

        #calculate the attention scores by matrix multiplying the queries with the transposed keys, only transpose on the data and not the batch (first position)
        attentionScores = queries @ keys.transpose(1,2)
        #mask the attention scores with a mask of which the upper diagonal filled is infinites
        maskedAttentionScores = attentionScores.masked_fill_(self.mask.bool()[:numberOfTokens,:numberOfTokens],-torch.inf)
        # normalized attention weights
        attentionWeights = torch.softmax(maskedAttentionScores / keys.shape[-1] ** 0.5,dim=-1)
        #apply the dropout mask to the masked and normalized weights
        attentionWeights = self.dropout(attentionWeights)
        print(attentionWeights)
        #calculate the context vector by matrix multiplying the masked attention weights with the values
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
batch = torch.stack((inputs,inputs),dim=0)
print(batch.shape)

torch.manual_seed(123)
sav1 = SelfAttentionV2(
    dimensionIn = 3,
    dimensionOut= 2,
    contextLength=batch.shape[1],
    dropoutP= 0.5)
contextVectors = sav1(batch)
print("tensor([0.3061, 0.8210])")
print(contextVectors)        