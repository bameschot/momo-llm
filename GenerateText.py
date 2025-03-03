import torch
import torch.nn as nn

from GPTModelConfig import *
from GPTModelStorage import *


def textToTokens(text,tkz):
    encoded = tkz.encode(text)
    return torch.tensor(encoded).unsqueeze(0)

def tokensToText(tokens,tkz):
    flat = tokens.squeeze(0)
    return tkz.decode(flat.tolist())

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

def generateText(model, idx, maxNewTokens,temperature=0.0,topK=None,eosId=None,printNextToken=False,tokenizer=None):
    contextSize = model.config[CONTEXT_LENGTH]
    for i in range(maxNewTokens):
        #ensure inx is no larget than the model supported max context size
        idxConditioned = idx[:,-contextSize:]

        #no_grad ensure that no backwards passes are performed when running the model, in this mode inference is performed efficiently
        with torch.no_grad():
            logits = model(idxConditioned)
        #turn batch,ntokens,ctxsize into batch,ctxsize to focus on the last step only
        logits = logits[:,-1,:]

        #topK filtering, if enables
        if topK is not None:
            topLogits,_ = torch.topk(logits,topK)
            #select the lowest probability still allowed by the topK selection
            minimumVal = topLogits[:,-1]
            #returns tensor elements as either their real value (if part of the topK range) or as negative infinity
            logits = torch.where(
                logits<minimumVal,
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )
            #apply temperature informed selection if applicable
            if(temperature>0.0):
                logits = logits/temperature
                probabilities = torch.softmax(logits,dim=-1) 
                idxNext = torch.multinomial(probabilities,num_samples=1)
        else:
            #turn the logits into probabilities
            #probabilities = torch.softmax(logits,dim=-1)
            #select the most probable next word, argmax returns the index of the higest probability which corresponds to a vocabulary index
            idxNext = torch.argmax(logits,dim=-1,keepdim=True)

        #if end of sequence token is detected end generation
        if idxNext == eosId:
            break

        #append the chosen token to the result
        idx = torch.cat((idx,idxNext),dim=1)
        if printNextToken and tokenizer is not None:
            print(tokensToText(tokens=idxNext,tkz=tokenizer))
    
    return idx

def generateTextShift(model, idx, maxNewTokens,temperature=0.0,topK=None,eosId=None,printNextToken=False,tokenizer=None):
    contextSize = model.config[CONTEXT_LENGTH]
    inputIdxLength = len(idx[0])

    #create an output buffer sized to accomodate the input context and the desired text to generate
    outIdx = torch.zeros(maxNewTokens,dtype=idx.dtype).to(device=idx.device)
    outIdx = torch.cat((idx,outIdx.unsqueeze(0)),dim=1)

    #inflate the input index to the context size and fill it with zeros, this will act as a ringbuffer
    zeroes = torch.zeros(contextSize-inputIdxLength,dtype=idx.dtype).to(device=idx.device)
    idx = torch.cat((zeroes.unsqueeze(0),idx),dim=1)
    
    for generatedTokenIdx in range(maxNewTokens):
        
        #Use index slicing to ensure that the prefix fill of the ringbuffer is ignored until the entire ringbuffer has been filled after which it can be used fully
        #slicingIndex = min(newTokenIdx+inputIdxLength,contextSize)*-1
        #idxConditioned = idx[:,slicingIndex:]
        if generatedTokenIdx+inputIdxLength<contextSize:
            #slicingIndex = min(newTokenIdx+inputIdxLength,contextSize)*-1
            idxConditioned = idx[:,(generatedTokenIdx+inputIdxLength)*-1:]
        else:
            idxConditioned = idx


        #no_grad ensure that no backwards passes are performed when running the model, in this mode inference is performed efficiently
        with torch.no_grad():
            logits = model(idxConditioned)
        #turn batch,ntokens,ctxsize into batch,ctxsize to focus on the last step only
        logits = logits[:,-1,:]

        #topK filtering, if enables
        if topK is not None:
            topLogits,_ = torch.topk(logits,topK)
            #select the lowest probability still allowed by the topK selection
            minimumVal = topLogits[:,-1]
            #returns tensor elements as either their real value (if part of the topK range) or as negative infinity
            logits = torch.where(
                logits<minimumVal,
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )
            #apply temperature informed selection if applicable
            if(temperature>0.0):
                logits = logits/temperature
                probabilities = torch.softmax(logits,dim=-1) 
                idxNext = torch.multinomial(probabilities,num_samples=1)
        else:
            #turn the logits into probabilities
            #probabilities = torch.softmax(logits,dim=-1)
            #select the most probable next word, argmax returns the index of the higest probability which corresponds to a vocabulary index
            idxNext = torch.argmax(logits,dim=-1,keepdim=True)

        #if end of sequence token is detected end generation
        if idxNext == eosId:
            break

        #append the chosen token to the result by shifting the idx tensor left and adding the new token to the last position
        idx = idx.roll(shifts=-1)
        idx[0,contextSize-1]=idxNext

        #Assign the new token to the output buffer
        outIdx[0,inputIdxLength+generatedTokenIdx]=idxNext

        if printNextToken and tokenizer is not None:
            print(tokensToText(tokens=idxNext,tkz=tokenizer),end="",flush=(generatedTokenIdx%3==0))
    
    return outIdx



###########
#
###########
# import tiktoken

# torch.set_printoptions(sci_mode=False)

# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#         device = torch.device("cpu")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")
#         device = torch.device("cpu")
# else:
#     device = torch.device("mps")

# #device = torch.device("cpu")
# print(f"Running on {device}!")

# model = GPTModel(config=GPT_CONFIG_SMALL).eval().to(device)
# #model = torch.compile(model)



        

