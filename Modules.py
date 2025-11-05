import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))
            )
        )

class FeedForward(nn.Module):
    def __init__(self,embeddingDimension, expansionFactor=4):
        super().__init__()
        self.layers = nn.Sequential(
            #expand
            nn.Linear(embeddingDimension,expansionFactor * embeddingDimension),
            #activate
            GELU(),
            #contract
            nn.Linear(4 * embeddingDimension, embeddingDimension)
        )

    def forward(self,x):
        return self.layers(x)
    
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
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        normalizedX = (x-mean) / torch.sqrt(var+self.epsilon)
        #scale and offset the normalized x with the trainable parameters
        return self.scale * normalizedX + self.shift
    
#
# Main attention module
#

class MultiHeadAttention(nn.Module):
    def __init__(self,numberOfHeads, embeddingDimension, contextLength,attentionDropoutRate,qkvBias=False):
        super().__init__()

        assert(embeddingDimension % numberOfHeads == 0), f"Output dimension must be divisable by the number of heads {embeddingDimension}/{numberOfHeads}"
        self.contextLength = contextLength
        self.embeddingDimension=embeddingDimension
        self.numberOfHeads = numberOfHeads 
        self.headDimension = embeddingDimension // numberOfHeads
        self.wQuery = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wKey = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wValue = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.dropout = nn.Dropout(attentionDropoutRate)
        self.outProjection = nn.Linear(embeddingDimension,embeddingDimension)        
        self.register_buffer('mask', torch.triu(torch.ones(contextLength,contextLength),diagonal=1))#,persistent=False)

        #cache
        self.register_buffer('cacheK',None,persistent=False)
        self.register_buffer('cacheV',None,persistent=False)
        self.pointerCurrentPosition = 0

        #gated attention
        self.wGate = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)

    def forward(self, x:torch.Tensor, useCache=False):
        #input data
        batchNr, numberOfTokens, dIn = x.shape

        #get the keys queries and values for each input
        queries = self.wQuery(x)
        keysNew = self.wKey(x)
        valuesNew = self.wValue(x)

        #initial attention gate value
        gate = self.wGate(x)

        #check for cache usage, if present use, if not register or ignore cache
        if useCache:
            if self.cacheK is None:
                self.cacheK, self.cacheV = keysNew, valuesNew
            else:
                self.cacheK = torch.cat([self.cacheK, keysNew], dim=1)
                self.cacheV = torch.cat([self.cacheV, valuesNew], dim=1)
            keys, values = self.cacheK, self.cacheV
        else:
            keys, values = keysNew, valuesNew

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

        #apply the mask on only the new tokens if cached
        numTokensQ = queries.shape[-2]
        numTokensK = keys.shape[-2]
        if useCache:
            maskBool = self.mask.bool()[self.pointerCurrentPosition:self.pointerCurrentPosition + numTokensQ, :numTokensK]
            self.pointerCurrentPosition += numTokensQ
        else:
            maskBool = self.mask.bool()[:numTokensQ, :numTokensK]

        #mask the attention scores with a mask of which the upper diagonal filled is infinites
        attentionScores.masked_fill_(maskBool,-torch.inf)
        #normalized attention weights
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1] ** 0.5,dim=-1)
        #apply the dropout mask to the masked and normalized weights
        attentionWeights = self.dropout(attentionWeights)
        
        #calculate the context vector by multipying the attention weights with the value and combine the head results
        contextVectors = (attentionWeights @ values).transpose(1,2)
        contextVectors = contextVectors.contiguous().view(batchNr, numberOfTokens,self.embeddingDimension)

        #apply gate
        contextVectors = contextVectors * torch.sigmoid(gate)
        
        # adds a linear projection
        return self.outProjection(contextVectors)
    
    def resetCache(self):
        self.cacheK, self.cacheV = None, None
        self.pointerCurrentPosition = 0 



#
#
# old self attention
#
#

class CausalSelfAttention (nn.Module):
    def __init__(self, embeddingDimension, contextLength,attentionDropoutRate,qkvBias=False):
        super().__init__()
        self.embeddingDimension=embeddingDimension
        self.wQuery = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wKey = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wValue = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.dropout = torch.nn.Dropout(attentionDropoutRate)
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
    def __init__(self,numberOfHeads, embeddingDimension, contextLength,attentionDropoutRate,qkvBias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalSelfAttention(embeddingDimension, contextLength,attentionDropoutRate,qkvBias) for _ in range(numberOfHeads)]
        )

    def forward(self, x:torch.Tensor):
        return torch.cat([head(x) for head in self.heads],dim=-1)
