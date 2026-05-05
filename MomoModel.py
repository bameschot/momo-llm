import torch
import torch.nn as nn

from MomoModelConfig import *

from Modules import RMSNormalization, FeedForwardBypass, MultiHeadAttention

class MomoModel(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        self.config=config
        dtType = getDataTypeFromConfig(config)
        
        #create the token embedding  
        self.tokenEmbeddings = nn.Embedding(config[VOCABULARY_SIZE],config[EMBEDDING_DIMENSION],dtype=dtType,device=device)
        
        #create the positional embedding unless configured to use NOPE encoding
        if not config.get(NOPE_ENCODING,False):
            self.positionalEmbeddings = nn.Embedding(config[CONTEXT_LENGTH],config[EMBEDDING_DIMENSION],dtype=dtType,device=device)
        else:
            self.positionalEmbeddings = None
        
        #create the dropout embedding
        if self.config[DROPOUT_EMBEDDING_RATE] > 0:
            self.dropoutEmbeddings = nn.Dropout(config[DROPOUT_EMBEDDING_RATE])
        else:
            self.dropoutEmbeddings = None
        #create the transformer blocks
        self.tranformerBlocks = nn.ModuleList(
            [MomoTransformerBlock(config,dtType=dtType,layer=layerDepth+1,device=device) for layerDepth in range(config[N_LAYERS])]
        )
        #create the final normalization layer
        self.finalNormalization = RMSNormalization(embeddingDimension=config[EMBEDDING_DIMENSION],layer=None,dtType=dtType,device=device)
        #create OutHead as a linear transformation
        self.outHead = nn.Linear(config[EMBEDDING_DIMENSION],config[VOCABULARY_SIZE],bias=False,dtype=dtType,device=device)

        self.currentPos = 0


    def forward(self,inIndex:torch.Tensor,useCache=False):
        #input tensor properties
        batchSize, sequenceLength = inIndex.shape
    
        #embeddings and positional embeddings for the input, ensures that the embeddings are on the correct device
        inTokenEmbeddings = self.tokenEmbeddings(inIndex)
        
        #add the positional embeddings if not using nope
        if self.positionalEmbeddings != None:
            if useCache:
                positionalIds = torch.arange(self.currentPos,self.currentPos+sequenceLength,device=inIndex.device,dtype=torch.int)
                self.currentPos+=sequenceLength
            else:
                positionalIds = torch.arange(0, sequenceLength, device=inIndex.device, dtype=torch.int)
            inPositionalEmbeddings = self.positionalEmbeddings(positionalIds).unsqueeze(0)

            #modify the embeddings with the positional embeddings
            x = inTokenEmbeddings + inPositionalEmbeddings
        else:
            x = inTokenEmbeddings
        
        #apply dropout
        if self.dropoutEmbeddings:
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


class MomoTransformerBlock(nn.Module):
    def __init__(self,config,layer,dtType,device):
        super().__init__()
        self.attention = MultiHeadAttention(
            config[N_HEADS], 
            config[EMBEDDING_DIMENSION], 
            config[CONTEXT_LENGTH],
            config[DROPOUT_ATTENTION_RATE],
            dtType,
            device,
            config[QKV_BIAS],
            config.get(GATED_ATTENTION,True)
        )
        self.feedForward = FeedForwardBypass(embeddingDimension=config[EMBEDDING_DIMENSION],dtType=dtType,device=device)
        self.normalizationLayer1 = RMSNormalization(embeddingDimension=config[EMBEDDING_DIMENSION],layer=None,dtType=dtType,device=device)
        self.normalizationLayer2 = RMSNormalization(embeddingDimension=config[EMBEDDING_DIMENSION],layer=layer,dtType=dtType,device=device)
        self.normalizationLayer3 = RMSNormalization(embeddingDimension=config[EMBEDDING_DIMENSION],layer=None,dtType=dtType,device=device)
        self.normalizationLayer4 = RMSNormalization(embeddingDimension=config[EMBEDDING_DIMENSION],layer=layer,dtType=dtType,device=device)

        
        if config[DROPOUT_SHORTCUT_RATE] > 0:
            self.dropoutShortcut = nn.Dropout(config[DROPOUT_SHORTCUT_RATE])
        else: 
            self.dropoutShortcut = None

    def forward(self,x, useCache=False):
        shortcut = x
        x = self.normalizationLayer1(x)
        x = self.attention(x,useCache)
        x = self.normalizationLayer2(x)

        if self.dropoutShortcut:
            x = self.dropoutShortcut(x)
        x = x + shortcut


        shortcut = x
        x = self.normalizationLayer3(x)
        x = self.feedForward(x)
        x = self.normalizationLayer4(x)

        if self.dropoutShortcut:
            x = self.dropoutShortcut(x)
        x = x + shortcut


        return x
    