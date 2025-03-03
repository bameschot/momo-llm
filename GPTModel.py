import torch
import torch.nn as nn

from GPTModelConfig import *

from Transformers import GPTTransformerBlock
from LayerNormalization import LayerNormalization

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
        self.tranformerBlocks = nn.Sequential(
            *[GPTTransformerBlock(config) for _ in range(config[N_LAYERS])]
        )
        #create the final normalization layer
        self.finalNormalization = LayerNormalization(embeddingDimension=config[EMBEDDING_DIMENSION])
        #create OutHead as a linear transformation
        self.outHead = nn.Linear(config[EMBEDDING_DIMENSION],config[VOCABULARY_SIZE],bias=False)

    def forward(self,inIndex:torch.Tensor):
        #input tensor properties
        batchSize, sequenceLength = inIndex.shape
        
        #embeddings and positional embeddings for the input, ensures that the embeddings are on the correct device
        inTokenEmbeddings = self.tokenEmbeddings(inIndex)
        inPositionalEmbeddings = self.positionalEmbeddings(
            torch.arange(sequenceLength,device=inIndex.device)
        )

        #modify the embeddings with the positional embeddings
        x = inTokenEmbeddings + inPositionalEmbeddings
        #apply dropout
        x = self.dropoutEmbeddings(x) 
        #apply the transformer blocks
        x = self.tranformerBlocks(x)
        #normalize the final result of the transformers
        x = self.finalNormalization(x)

        #produce output logits with the linear layer
        logits = self.outHead(x)
        return logits 
    
    def numberOfParameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def memSizeMb(self):
        sizeBytes = self.numberOfParameters() * 4 #assumes float32
        return sizeBytes / (1024 * 1024)



