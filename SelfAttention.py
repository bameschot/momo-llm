import torch
import torch.nn as nn

class CausalSelfAttention (nn.Module):
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

class CausalSelfAttention (nn.Module):
    def __init__(self,dimensionIn,dimensionOut,contextLength,dropoutP,keyValueBias=False):
        super().__init__()
        self.dimensionOut=dimensionOut
        self.wQuery = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wKey = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wValue = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.dropout = torch.nn.Dropout(dropoutP)
        self.register_buffer('mask', torch.triu(torch.ones(contextLength,contextLength),diagonal=1))


    def forward(self, x:torch.Tensor):
        #input data
        batchNr, numberOfTokens, dIn = x.shape

        #get the keys queries and values for each input
        queries = self.wQuery(x)
        keys = self.wKey(x)
        values = self.wValue(x)

        #calculate the attention scores by matrix multiplying the queries with the transposed keys, only transpose on the data and not the batch (first position)
        attentionScores = queries @ keys.transpose(1,2)
        #mask the attention scores with a mask of which the upper diagonal filled is infinites
        attentionScores.masked_fill_(self.mask.bool()[:numberOfTokens,:numberOfTokens],-torch.inf)
        # normalized attention weights
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1] ** 0.5,dim=-1)
        #apply the dropout mask to the masked and normalized weights
        attentionWeights = self.dropout(attentionWeights)
        #calculate the context vector by matrix multiplying the masked attention weights with the values
        contextVectors = attentionWeights @ values
        return contextVectors

class MultiHeadCausalAttentionWrapper(torch.nn.Module):
    def __init__(self,dimensionIn,dimensionOut,contextLength,dropoutP,numberOfHeads,keyValueBias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalSelfAttention(
                dimensionIn=dimensionIn,
                dimensionOut=dimensionOut,
                contextLength=contextLength,
                dropoutP=dropoutP,
                keyValueBias=keyValueBias
            ) for _ in range(numberOfHeads)]
        )

    def forward(self, x:torch.Tensor):
        return torch.cat([head(x) for head in self.heads],dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self,dimensionIn,dimensionOut,contextLength,dropoutP,numberOfHeads,keyValueBias=False):
        super().__init__()

        assert(dimensionOut % numberOfHeads == 0), f"Output dimension must be divisable by the number of heads {dimensionOut}/{numberOfHeads}"

        self.dimensionOut=dimensionOut
        self.numberOfHeads = numberOfHeads 
        self.headDimension = dimensionOut // numberOfHeads
        self.wQuery = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wKey = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.wValue = nn.Linear(dimensionIn,dimensionOut,keyValueBias)
        self.dropout = nn.Dropout(dropoutP)
        self.outProjection = nn.Linear(dimensionOut,dimensionOut)        
        self.register_buffer('mask', torch.triu(torch.ones(contextLength,contextLength),diagonal=1))

    def forward(self, x:torch.Tensor):
        #input data
        batchNr, numberOfTokens, dIn = x.shape

        #get the keys queries and values for each input
        queries = self.wQuery(x)
        keys = self.wKey(x)
        values = self.wValue(x)

        #create views for each of the heads
        queries = queries.view(batchNr, numberOfTokens, self.numberOfHeads, self.headDimension)
        keys = keys.view(batchNr, numberOfTokens, self.numberOfHeads, self.headDimension)
        values = values.view(batchNr, numberOfTokens, self.numberOfHeads, self.headDimension)

        #transpose to cahnge from batch,numberOfTokens,numberOfHeads,headDimension -> batch,numberOfHeads,numberOfTokens,headDimension
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        #determine the attention scores by calculating the dot product for each head
        attentionScores = queries @ keys.transpose(2,3)
        #mask the attention scores with a mask of which the upper diagonal filled is infinites
        attentionScores.masked_fill_(self.mask.bool()[:numberOfTokens,:numberOfTokens],-torch.inf)
        #normalized attention weights
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1] ** 0.5,dim=-1)
        #apply the dropout mask to the masked and normalized weights
        attentionWeights = self.dropout(attentionWeights)
        
        #calculate the context vector by multipying the attention weights with the value and combine the head results
        contextVectors = (attentionWeights @ values).transpose(1,2)
        contextVectors = contextVectors.contiguous().view(batchNr, numberOfTokens,self.dimensionOut)

        # adds a linear projection
        return self.outProjection(contextVectors)



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
sav1 = MultiHeadAttention(
    dimensionIn = 3,
    dimensionOut= 2,
    contextLength=batch.shape[1],
    numberOfHeads= 2,
    dropoutP= 0.0)
contextVectors = sav1(batch)
# tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]],

#         [[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)

# tensor([[[-0.5740,  0.2216],
#          [-0.7320,  0.0155],
#          [-0.7774, -0.0546],
#          [-0.6979, -0.0817],
#          [-0.6538, -0.0957],
#          [-0.6424, -0.1065]],

#         [[-0.5740,  0.2216],
#          [-0.7320,  0.0155],
#          [-0.7774, -0.0546],
#          [-0.6979, -0.0817],
#          [-0.6538, -0.0957],
#          [-0.6424, -0.1065]]], grad_fn=<CatBackward0>)

print(contextVectors)        
