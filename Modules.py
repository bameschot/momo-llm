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

class MultiHeadAttention(nn.Module):
    def __init__(self,numberOfHeads, embeddingDimension, contextLength,attentionDropoutRate,qkvBias=False):
        super().__init__()

        assert(embeddingDimension % numberOfHeads == 0), f"Output dimension must be divisable by the number of heads {embeddingDimension}/{numberOfHeads}"

        self.embeddingDimension=embeddingDimension
        self.numberOfHeads = numberOfHeads 
        self.headDimension = embeddingDimension // numberOfHeads
        self.wQuery = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wKey = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.wValue = nn.Linear(embeddingDimension,embeddingDimension,qkvBias)
        self.dropout = nn.Dropout(attentionDropoutRate)
        self.outProjection = nn.Linear(embeddingDimension,embeddingDimension)        
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
        contextVectors = contextVectors.contiguous().view(batchNr, numberOfTokens,self.embeddingDimension)

        # adds a linear projection
        return self.outProjection(contextVectors)  

#####
#adapted from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
#####
class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, numberOfHeads, embeddingDimension, contextLength,attentionDropoutRate,qkvBias=False):
        super().__init__()

        assert(embeddingDimension % numberOfHeads == 0), f"Output dimension must be divisable by the number of heads {embeddingDimension}/{numberOfHeads}"

        self.numberOfHeads = numberOfHeads
        self.context_length = contextLength
        self.headDimension = embeddingDimension // numberOfHeads
        self.embeddingDimension = embeddingDimension

        self.qkv = nn.Linear(embeddingDimension, 3 * embeddingDimension, bias=qkvBias)
        self.proj = nn.Linear(embeddingDimension, embeddingDimension)
        self.dropout = attentionDropoutRate

    def forward(self, x):
        batchSize, numberOfTokens, embeddingDimension = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batchSize, numberOfTokens, 3, self.numberOfHeads, self.headDimension)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        useDropout = 0. if not self.training else self.dropout

        contextVector = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=useDropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        contextVector = contextVector.transpose(1, 2).contiguous().view(batchSize, numberOfTokens, self.embeddingDimension)

        contextVector = self.proj(contextVector)

        return contextVector