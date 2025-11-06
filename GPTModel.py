import torch
import torch.nn as nn

from GPTModelConfig import *

from Modules import LayerNormalization, FeedForward, MultiHeadAttention

class GPTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        #create the token embedding  
        self.tokenEmbeddings = nn.Embedding(config[VOCABULARY_SIZE],config[EMBEDDING_DIMENSION])
        #create the positional embedding
        self.positionalEmbeddings = nn.Embedding(config[CONTEXT_LENGTH],config[EMBEDDING_DIMENSION])
        #create the dropout embedding
        self.dropoutEmbeddings = nn.Dropout(config[DROPOUT_EMBEDDING_RATE])
        #create the transformer blocks
        self.tranformerBlocks = nn.ModuleList(
            [GPTTransformerBlock(config) for _ in range(config[N_LAYERS])]
        )
        #create the final normalization layer
        self.finalNormalization = LayerNormalization(embeddingDimension=config[EMBEDDING_DIMENSION])
        #create OutHead as a linear transformation
        self.outHead = nn.Linear(config[EMBEDDING_DIMENSION],config[VOCABULARY_SIZE],bias=False)

        self.currentPos = 0


    def forward(self,inIndex:torch.Tensor,useCache=False):
        #input tensor properties
        batchSize, sequenceLength = inIndex.shape
    
        #embeddings and positional embeddings for the input, ensures that the embeddings are on the correct device
        inTokenEmbeddings = self.tokenEmbeddings(inIndex)
        if useCache:
            positionalIds = torch.arange(self.currentPos,self.currentPos+sequenceLength,device=inIndex.device,dtype=torch.long)
            self.currentPos+=sequenceLength
        else:
            positionalIds = torch.arange(0, sequenceLength, device=inIndex.device, dtype=torch.long)
        inPositionalEmbeddings = self.positionalEmbeddings(positionalIds).unsqueeze(0)

        #modify the embeddings with the positional embeddings
        x = inTokenEmbeddings + inPositionalEmbeddings
        
        #apply dropout
        x = self.dropoutEmbeddings(x) 
        
        #apply the transformer blocks
        for blk in self.tranformerBlocks:
            x = blk(x,useCache)
        
        #normalize the final result of the transformers
        x = self.finalNormalization(x)

        #produce output logits with the linear layer
        logits = self.outHead(x)
        return logits 
    
    def numberOfParameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def memSizeMb(self):
        sizeBytes = self.numberOfParameters() * 2 #assumes float6 or bfloat16
        return round(sizeBytes / (1024 * 1024),2)
    
    def resetCache(self):
        self.currentPos = 0
        for block in self.tranformerBlocks:
            block.attention.resetCache()


class GPTTransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config[N_HEADS], 
            config[EMBEDDING_DIMENSION], 
            config[CONTEXT_LENGTH],
            config[DROPOUT_ATTENTION_RATE],
            config[QKV_BIAS]
        )
        self.feedForward = FeedForward(embeddingDimension=config[EMBEDDING_DIMENSION])
        self.normalizationLayer1 = LayerNormalization(embeddingDimension=config[EMBEDDING_DIMENSION])
        self.normalizationLayer2 = LayerNormalization(embeddingDimension=config[EMBEDDING_DIMENSION])
        self.dropoutShortcut = nn.Dropout(config[DROPOUT_SHORTCUT_RATE])

    def forward(self,x, useCache=False):
        shortcut = x
        x = self.normalizationLayer1(x)
        x = self.attention(x,useCache)
        x = self.dropoutShortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.normalizationLayer2(x)
        x = self.feedForward(x)
        x = self.dropoutShortcut(x)
        x = x + shortcut

        return x
    