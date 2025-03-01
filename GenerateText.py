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
        print(idxConditioned.device)
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
        idx = torch.cat((idx,idxNext),dim=1)
    
    return idx

###########
#
###########
import tiktoken

torch.manual_seed(123)
torch.set_printoptions(sci_mode=False)

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        device = torch.device("cpu")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device("cpu")
else:
    print("Running on MPS!")
    device = torch.device("mps")

device = torch.device("cpu")

model = GPTModel(GPT_CONFIG_SMALL)
model.to(device)
model = torch.compile(model)

#disables dropout layers for faster and better inference
model.eval()

tokenizer = tiktoken.get_encoding('gpt2')
startContext = "Hello, I am"
encoded = tokenizer.encode(startContext)
print(f"encoded: {encoded}")
encodedTensor = torch.tensor(encoded,device=device).unsqueeze(0)

print(f"encodedTensor: {encodedTensor} / {encodedTensor.shape}")

requestedOutputSize = 200
print("Start inference")
startTs = time.time() * 1000.0
generatedOutput = simpleTextGeneration(model=model,idx=encodedTensor,maxNewTokens=requestedOutputSize)
endTs = time.time() * 1000.0
runtimeMs = (endTs-startTs)

print(f"Generated output: {generatedOutput}")
print(f"Generated shape: {generatedOutput.shape}")


decodedText = tokenizer.decode(generatedOutput.squeeze(0).tolist())
print(f"It Lives!: {decodedText}")
print(f"Runtime: {runtimeMs} ms. T/Ps: {(requestedOutputSize/(runtimeMs/1000.0))}")





        

