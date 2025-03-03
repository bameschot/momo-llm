import time

import torch

from Preprocessing import GPT2Tokenizer
from GenerateText import generateText,generateTextShift, textToTokens, tokensToText
from GPTModel import GPTModel
from GPTModelConfig import *
from GPTModelStorage import *


input = "Theory of labour"
tokensToGenerate = 1500

device = torch.device("mps:0")
modelname = "TestEconomy"

tokenizer = GPT2Tokenizer()
model = loadModel(
    modelName=modelname,
    config=GPT_CONFIG_SMALL,
    device=device
)

# model = GPTModel(GPT_CONFIG_SMALL).to(device)
# model.eval()

print(f"Loaded model: {modelname}, running on device {device}")


inputTokens = textToTokens(input,tokenizer).to(device)

print("-----------------------")
print("start generating")
print("-----------------------")
startTs = time.time() * 1000.0

with torch.no_grad():
    outputTokens = generateTextShift(
        model=model,
        idx=inputTokens,
        maxNewTokens=tokensToGenerate,
        temperature=0.2,
        topK=5,
        printNextToken=True,
        tokenizer=tokenizer)

outputText = tokensToText(outputTokens,tokenizer)
endTs = time.time() * 1000.0
runtimeMs = (endTs-startTs)

print("-----------------------")
print("Generated")
print("-----------------------")
print(outputText)
print(f"Runtime: {runtimeMs} ms. T/Ps: {(tokensToGenerate/(runtimeMs/1000.0))}")
