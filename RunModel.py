import time
import argparse

import torch
import torch._dynamo

from Tokenizers import *
from GenerateText import generateTextShift,generateText, textToTokens, tokensToText
from GPTModelConfig import *
from GPTModelStorage import *


########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Runs a GPT model"
    )
parser.add_argument("--modelname", type=str,default="TestEconomy_small_r", help="The name of the model to run, model must be present in./models/<model-name>/<model-name>.model")
parser.add_argument("--tokenizer", type=str,default="gpt2", help="The name of the tokenizer to parse input and output in")
parser.add_argument("--tokenizerVocabFile", type=str,default=None, help="The path of the vocabulary or model file to load for the tokenizer")
parser.add_argument("--prompt", type=str,default="Theory of labour", help="The input promt to start generating text with")
parser.add_argument("--forceLowerCase", action='store_true', help="ensures that all text becomes lowercase")
parser.add_argument("--tokensToGenerate", type=int, default=100, help="The number of tokens to generate")
parser.add_argument("--temperature",type=float, default=0.5,help="The temperature to use when generating tokens, regulates variety in output")
parser.add_argument("--topK",type=int, default=50,help="The numper of most probable to pick the next token from, together with temperature this regulates variety in ouput")
parser.add_argument("--printNextToken", action='store_true',help="Indicates if the each token should be printed as it is generated")
parser.add_argument("--compileModel", action='store_true',help="Indicates if the model should be compiled first")
parser.add_argument('--device', type=str, default=None, help= "indicates the device the model has to run on, if not provided the system autodetects in the order cuda->mps->cpu")

args = parser.parse_args()

p_modelname = args.modelname
p_tokenizer = args.tokenizer
p_tokenizerVocabFile = args.tokenizerVocabFile
p_prompt = args.prompt 
p_forceLowerCase = args.forceLowerCase
p_tokensToGenerate = args.tokensToGenerate
p_temperature = args.temperature     
p_topK =args.topK
p_printNextToken =args.printNextToken
p_compileModel = args.compileModel
p_device = args.device


########################################
#Script
########################################

if p_device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(p_device)


model = loadModel(
    modelName=p_modelname,
    device=device
)

tokenizer = initializeTokenizer(model.config[TOKENIZER_TYPE],model.config[TOKENIZER_NAME])

print(f"Model loaded: {p_modelname}, running on device {device} with {model.numberOfParameters():_} parameters and memory size: {model.memSizeMb():_} mb")

if p_compileModel:
    print("compiling model")
    #required for mps compat
    if torch.mps.is_available():
        torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, mode="max-autotune")

prompt = p_prompt.lower() if p_forceLowerCase else p_prompt
inputTokens = textToTokens(prompt,tokenizer).to(device)

print("-----------------------")
print("start generating")
print("-----------------------")
print(f"Prompt: {prompt} [{inputTokens.shape[1]} tokens]")

startTs = time.time() * 1000.0

with torch.no_grad():
    outputTokens = generateText(
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

print("")
print("-----------------------")
print("Generated")
print("-----------------------")
print(outputText)
print(f"Runtime: {runtimeMs} ms. T/Ps: {(p_tokensToGenerate/(runtimeMs/1000.0))} [{outputTokens.shape[1]} tokens]")
