import torch
import torch.nn as nn

VOCABULARY_SIZE = "VocabularySize"
CONTEXT_LENGTH = "ContextLength"
EMBEDDING_DIMENSION = "EmbeddingDimension"
N_HEADS = "NumberOfHeads"
N_LAYERS = "NumberOfLayers"
DROP_RATE = "DropRate"
QKV_BIAS = "QKVBias"

GPT_CONFIG_124M = {
    VOCABULARY_SIZE: 50257,
    CONTEXT_LENGTH: 1024,
    EMBEDDING_DIMENSION: 768,
    N_HEADS: 12,
    N_LAYERS: 12,
    DROP_RATE: 0.1,
    QKV_BIAS: False,
}

class GPTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        #create the token embedding  
        self.tokenEmbeddings = nn.Embedding(config[VOCABULARY_SIZE],config[EMBEDDING_DIMENSION])
        #create the positional embedding
        self.positionalEmbeddings = nn.Embedding(config[CONTEXT_LENGTH],config[EMBEDDING_DIMENSION])
        #create the dropout embedding
        self.dropoutEmbeddings = nn.Dropout(config[DROP_RATE])
        #create the transformer blocks
        self.tranformerBlock = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config[N_LAYERS])]
        )
        #create the final normalization layer
        self.finalNormalization = DummyLayerNormalization(normalizedShape=config[EMBEDDING_DIMENSION])
        #create OutHead as a linear transformation
        self.outHead = nn.Linear(config[EMBEDDING_DIMENSION],config[VOCABULARY_SIZE],bias=False)

    def forward(self,inIndex:torch.Tensor):
        #input tensor properties
        batchSize, sequenceLength = inIndex.shape
        
        #embeddings and positional embeddings for the input
        inTokenEmbeddings = self.tokenEmbeddings(inIndex)
        inPositionalEmbeddings = self.positionalEmbeddings(
            torch.arange(sequenceLength,device=inIndex.device)
        )

        #modify the embeddings with the positional embeddings
        x = inTokenEmbeddings + inPositionalEmbeddings
        #apply dropout
        x = self.dropoutEmbeddings(x) 
        #apply the transformer blocks
        x = self.tranformerBlock(x)
        #normalize the final result of the transformers
        x = self.finalNormalization(x)

        #produce output logits with the linear layer
        logits = self.outHead(x)
        return logits 

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self,x):
        return x

class DummyLayerNormalization(nn.Module):
    def __init__(self, normalizedShape, epsilon=1e-5):
        super().__init__()
    
    def forward(self,x):
        return x






import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch,dim=0)

print(batch)

torch.manual_seed(123)
gptModel = GPTModel(GPT_CONFIG_124M)
logits = gptModel(batch)

print(f"Output shape: {logits.shape}")
print(f"Output: {logits}")
