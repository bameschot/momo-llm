import torch
import torch.nn as nn

from GPTModelConfig import *

class CausalSelfAttention (nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embeddingDimension=config[EMBEDDING_DIMENSION]
        self.wQuery = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.wKey = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.wValue = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.dropout = torch.nn.Dropout(config[DROPOUT_ATTENTION_RATE])
        self.register_buffer('mask', torch.triu(torch.ones(config[CONTEXT_LENGTH],config[CONTEXT_LENGTH]),diagonal=1))


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
    def __init__(self,config):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalSelfAttention(config) for _ in range(config[N_HEADS])]
        )

    def forward(self, x:torch.Tensor):
        return torch.cat([head(x) for head in self.heads],dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()

        assert(config[EMBEDDING_DIMENSION] % config[N_HEADS] == 0), f"Output dimension must be divisable by the number of heads {config[EMBEDDING_DIMENSION]}/{config[N_HEADS]}"

        self.embeddingDimension=config[EMBEDDING_DIMENSION]
        self.numberOfHeads = config[N_HEADS] 
        self.headDimension = config[EMBEDDING_DIMENSION] // config[N_HEADS]
        self.wQuery = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.wKey = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.wValue = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION],config[QKV_BIAS])
        self.dropout = nn.Dropout(config[DROPOUT_ATTENTION_RATE])
        self.outProjection = nn.Linear(config[EMBEDDING_DIMENSION],config[EMBEDDING_DIMENSION])        
        self.register_buffer('mask', torch.triu(torch.ones(config[CONTEXT_LENGTH],config[CONTEXT_LENGTH]),diagonal=1))

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
        contextVectors = contextVectors.contiguous().view(batchNr, numberOfTokens,self.embeddingDimension)

        # adds a linear projection
        return self.outProjection(contextVectors)  
