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

def generateText(model, idx, maxNewTokens,temperature=0.9,topK=40,eosId=3,printNextToken=False,tokenizer=None):
    contextSize = model.config[CONTEXT_LENGTH]
    for i in range(maxNewTokens):
        #ensure idx is no larger than the model supported max context size
        idxConditioned = idx[:,-contextSize:]

        #no_grad ensure that no backwards passes are performed when running the model, in this mode inference is performed efficiently
        with torch.no_grad():
            logits = model(idxConditioned)
        #turn batch,ntokens,ctxsize into batch,ctxsize to focus on the last step only
        logits = logits[:,-1,:]

        #topK filtering, if enabled
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
            print(tokensToText(tokens=idxNext,tkz=tokenizer),end="",flush=True)
    
    return idx


def generateTextCached(model, idx, maxNewTokens,temperature=0.9,topK=40,eosId=None,printNextToken=False,tokenizer=None):
    contextSize = model.config[CONTEXT_LENGTH]
    #no_grad ensure that no backwards passes are performed when running the model, in this mode inference is performed efficiently
    with torch.no_grad():
        
        # start the counter to keep track of the size of the text generated or started with so far
        epochTextGenerated = len(idx[0])

        #ensure idx is no larger than the model supported max context size and fill the cache with the prompt   
        model.resetCache()
        logits = model(idx[:,-contextSize:],True)

        #generate the requested logits and (on the first pass process the first generated logit of the initial model call)
        for i in range(maxNewTokens):

            #turn batch,ntokens,ctxsize into batch,ctxsize to focus on the last step only
            logits = logits[:,-1,:]

            #topK filtering, if enabled
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
                print(tokensToText(tokens=idxNext,tkz=tokenizer),end="",flush=True)
            
            # if the generated text exceeds this generation epoch the context size reset the model and feed it half the previously generated context to start with
            # this prevents the model from generating errors when exceeding the context size
            if(epochTextGenerated>0 and epochTextGenerated % contextSize==0):
                
                #take half of the previously generated text and reset the epoch text generated counter to the provided restart prompt size
                newGenIdx = idx[:,-int(epochTextGenerated-(contextSize/2)):]
                epochTextGenerated = len(newGenIdx[0])

                # reset the cache and generate further based on the last half of the text
                model.resetCache()
                logits = model(newGenIdx)
            else: 
                # generate the next logit using only the last generated idx as input relying on the cache for the previous results
                logits = model(idxNext) 
                epochTextGenerated += 1

        
    return idx
