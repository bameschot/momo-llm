import tiktoken
import torch
import torch.nn as nn

from GPTModel import GPTModel
from GPTModelConfig import *
#from main import GPTModel, GPT_CONFIG_124M

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

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)

# startContext = "Every effort moves you"

# outputTokens = simpleTextGeneration(
#     model,
#     textToTokens(startContext,tokenizer),
#     20
# )

# print(tokensToText(outputTokens,tokenizer))

inputs = torch.tensor([
    [16833,3626,6100],
    [40,1107,588]]
    )

targets = torch.tensor([
    [3626,6100,345],
    [1107,588, 11311]]
)

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits,dim=-1)
print(probas.shape)

tokenIds = torch.argmax(probas,dim=-1,keepdim=True)
print(tokenIds)

print(f"input b1: {tokensToText(inputs[0],tokenizer)}")
print(f"target b1: {tokensToText(targets[0],tokenizer)}")
print(f"output b1: {tokensToText(tokenIds[0].flatten(),tokenizer)}")

tIdx = 0
targetProbs1 = probas[tIdx,[0,1,2],targets[tIdx]]
print(f"text {tIdx}: {targetProbs1}") 

tIdx = 1
targetProbs2 = probas[tIdx,[0,1,2],targets[tIdx]]
print(f"text {tIdx}: {targetProbs2}") 

logProbs = torch.log(torch.cat((targetProbs1,targetProbs2)))
print(f"log probabilities: {logProbs}")
avgLogProbs = torch.mean(logProbs) * -1
print(f"negative average log probabilities: {avgLogProbs}")

flattenedLogits = logits.flatten(0,1)
flattenedTarget = targets.flatten()

loss = nn.functional.cross_entropy(flattenedLogits,flattenedTarget)
print(f"loss: {loss}")
perplexity = torch.exp(loss)
print(f"perplexity: {perplexity}")