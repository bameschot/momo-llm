import time
import argparse

import torch
import torch._dynamo

from Preprocessing import GPT2Tokenizer
from GenerateText import generateText,generateTextShift, textToTokens, tokensToText
from GPTModel import GPTModel
from GPTModelConfig import *
from GPTModelStorage import *

#Makes torch mps compiling work on MAC!
torch._dynamo.config.suppress_errors = True

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Runs a GPT model"
    )
parser.add_argument("--modelname", type=str,default="TestEconomy_small_r", help="The name of the model to run, model must be present in./models/<model-name>/<model-name>.model")
parser.add_argument("--prompt", type=str,default="Theory of labour", help="The input promt to start generating text with")
parser.add_argument("--tokensToGenerate", type=int, default=100, help="The number of tokens to generate")
parser.add_argument("--temperature",type=float, default=0.2,help="The temperature to use when generating tokens, regulates variety in output")
parser.add_argument("--topK",type=int, default=5,help="The numper of most probable to pick the next token from, together with temperature this regulates variety in ouput")
parser.add_argument("--printNextToken", action='store_true',help="Indicates if the each token should be printed as it is generated")
parser.add_argument("--compileModel", action='store_true',help="Indicates if the model should be compiled first")
args = parser.parse_args()

p_modelname = args.modelname
p_prompt = args.prompt 
p_tokensToGenerate = args.tokensToGenerate
p_temperature= args.temperature     
p_topK=args.topK
p_printNextToken=args.printNextToken
p_compileModel = args.compileModel

########################################
#Script
########################################

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


tokenizer = GPT2Tokenizer()
model = loadModel(
    modelName=p_modelname,
    device=device
)
print(f"Model loaded: {p_modelname}, running on device {device}")

if p_compileModel:
    model = torch.compile(model, mode="reduce-overhead",backend="aot_eager")

inputTokens = textToTokens(p_prompt,tokenizer).to(device)

print("-----------------------")
print("start generating")
print("-----------------------")
print(f"Prompt: {p_prompt}")

startTs = time.time() * 1000.0

with torch.no_grad():
    outputTokens = generateTextShift(
        model=model,
        idx=inputTokens,
        maxNewTokens=p_tokensToGenerate,
        temperature=p_temperature,
        topK=p_topK,
        printNextToken=True,
        tokenizer=tokenizer)

outputText = tokensToText(outputTokens,tokenizer)
endTs = time.time() * 1000.0
runtimeMs = (endTs-startTs)

print("-----------------------")
print("Generated")
print("-----------------------")
print(outputText)
print(f"Runtime: {runtimeMs} ms. T/Ps: {(p_tokensToGenerate/(runtimeMs/1000.0))}")
