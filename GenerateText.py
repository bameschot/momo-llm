import time

import torch
import torch.nn as nn
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
        #turn the logits into probabilities
        probabilities = torch.softmax(logits,dim=-1)
        #select the most probable next word, argmax returns the index of the higest probability which corresponds to a vocabulary index
        idxNext = torch.argmax(probabilities,dim=-1,keepdim=True)

        #append the chosen token to the result
        idx = torch.cat((idx,idxNext),dim=1)
    
    return idx

def generateText(model, idx, maxNewTokens,temperature=0.0,topK=None,eosId=None):
    contextSize = model.config[CONTEXT_LENGTH]
    for _ in range(maxNewTokens):
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
    
    return idx

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

# #disables dropout layers for faster and better inference

# tokenizer = tiktoken.get_encoding('gpt2')
# startContext = "Hello, I am"
# encoded = tokenizer.encode(startContext)
# print(f"encoded: {encoded}")
# encodedTensor = torch.tensor(encoded,device=device).unsqueeze(0).to(device)

# print(f"encodedTensor: {encodedTensor} / {encodedTensor.shape}")

# requestedOutputSize = 256
# print("Start inference")
# startTs = time.time() * 1000.0
# with torch.no_grad():
#     generatedOutput = generateText(model=model,idx=encodedTensor,maxNewTokens=requestedOutputSize,temperature=2,topK=5)
# endTs = time.time() * 1000.0
# runtimeMs = (endTs-startTs)

# print(f"Generated output: {generatedOutput}")
# print(f"Generated shape: {generatedOutput.shape}")


# decodedText = tokenizer.decode(generatedOutput.squeeze(0).tolist())
# print(f"It Lives!: {decodedText}")
# print(f"Runtime: {runtimeMs} ms. T/Ps: {(requestedOutputSize/(runtimeMs/1000.0))}")





        

