import tiktoken
import torch

from GPTModel import GPTModel
from GPTModelConfig import *

def simpleTextGeneration(model, idx, maxNewTokens):
    contextSize = model.config[CONTEXT_LENGTH]
    for _ in range(maxNewTokens):
        #ensure inx is no larget than the model supported max context size
        idxConditioned = idx[:,-contextSize:]

        #no_grad ensure that no backwards passes are performed when running the model, in this mode inference is performed efficiently
        with torch.no_grad():
            logits = model(idxConditioned)
        #turn batch,ntokens,ctxsize into batch,ctxsize to focus on the last step only
        logits = logits[:,-1,:]
        #turn the logits into probabilities
        probabilities = torch.softmax(logits,dim=-1)
        #select the most probable next word, argmax returns the index of the higest probability which corresponds to a vocabulary index
        idxNext = torch.argmax(probabilities,dim=-1,keepdim=True)

        #append the chosen token to the result
        idx = torch.cat((idx,idxNext),dim=1)
    
    return idx

def textToTokens(text,tkz):
    encoded = tkz.encode(text,allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def tokensToText(tokens,tkz):
    flat = tokens.squeeze(0)
    return tkz.decode(flat.tolist())

tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()

startContext = "Every effort moves you"

outputTokens = simpleTextGeneration(
    model,
    textToTokens(startContext,tokenizer),
    20
)

print(tokensToText(outputTokens,tokenizer))
