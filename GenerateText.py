import torch
from GPTModelConfig import *
from GPTModel import GPTModel

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
        #turn the logits into probailities
        probabilities = torch.softmax(logits,dim=-1)
        #select the most probable next word, argmax returns the index of the higest probability which corresponds to a vocabulary index
        idxNext = torch.argmax(probabilities,dim=-1,keepdim=True)
        #append the chosen token to the result
        idx = torch.cat((id,idxNext),dim=1)
    
    return idx


        

