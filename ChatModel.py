import time
import argparse

import torch
import torch._dynamo

from Tokenizers import *
from GenerateText import generateText, generateTextCached, textToTokens, tokensToText
from GPTModelConfig import *
from GPTModelStorage import *


########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Runs a chat with a momo-llm model"
    )
parser.add_argument("--model", type=str,default="TestEconomy_small_r", help="The name of the model to run, model must be present in./models/<model-name>/<model-name>.model")
parser.add_argument("--forceLowerCase", action='store_true', help="ensures that all text becomes lowercase")
parser.add_argument("--autoAppendPeriod", action='store_true', help="appends a period to each chat if missing and no other sign is present")
parser.add_argument("--tokensToGenerate", type=int, default=100, help="The number of tokens to generate")
parser.add_argument("--temperature",type=float, default=0.5,help="The temperature to use when generating tokens, regulates variety in output")
parser.add_argument("--topK",type=int, default=-1,help="The numper of most probable to pick the next token from, together with temperature this regulates variety in ouput, use < 0 to turn off")
parser.add_argument("--minP",type=float, default=0.1,help="Regularizes the cutoff point of tokens based on the certainty of the models output, disabled if topK is used")
parser.add_argument("--printNextToken", action='store_true',help="Indicates if the each token should be printed as it is generated")
parser.add_argument("--compileModel", action='store_true',help="Indicates if the model should be compiled first")
parser.add_argument("--ommitEos", action='store_true',help="Indicates if the end of sequence should not be appended to each turn in the chat")
parser.add_argument('--device', type=str, default=None, help= "indicates the device the model has to run on, if not provided the system autodetects in the order cuda->mps->cpu")

args = parser.parse_args()

p_model = args.model
p_autoAppendPeriod = args.autoAppendPeriod
p_forceLowerCase = args.forceLowerCase
p_tokensToGenerate = args.tokensToGenerate
p_temperature = args.temperature     
p_topK =args.topK
p_minP =args.minP
p_printNextToken =args.printNextToken
p_compileModel = args.compileModel
p_ommitEos = args.ommitEos
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


model,spModelBytes = loadModel(
    modelName=p_model,
    device=device
)

if spModelBytes is None:
    tokenizer = initializeTokenizer(model.config[TOKENIZER_TYPE],model.config[TOKENIZER_NAME])
else: 
    tokenizer = initializeTokenizerFromModelBytes(model.config[TOKENIZER_TYPE],p_model, spModelBytes)

print(f"Model loaded: {p_model}, running on device {device} with {model.numberOfParameters():_} parameters and memory size: {model.memSizeMb():_} mb")

if p_compileModel:
    print("compiling model")
    #required for mps compat
    if torch.mps.is_available():
        torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, mode="max-autotune")

conversationHistoryTxt = ""
conversationHistoryTokens = torch.empty(size=(1,0),dtype=torch.int32)
totalRunTimeMs = 0
numOutputTokens = 0
numInputTokens = 0
contextWindow = model.config[CONTEXT_LENGTH]

inputTxt = ''
while True:
    #request input
    print("> ", end=" ")
    inputTxt = input()
    
    if inputTxt == '':
        inputTxt = '.'

    if inputTxt == '/bye':
        break

    if inputTxt == '/model':
        print(print(f"model: {p_model} with {model.numberOfParameters():_} parameters and memory size: {model.memSizeMb():_} mb running on device {device} config: {model.config}"))
        continue

    # parse the input and append eos and period if required, add to the conversation history
    inputTxt = inputTxt+"." if p_autoAppendPeriod and inputTxt[-1] not in ['!','?'] else inputTxt
    inputTxt = inputTxt if p_ommitEos else inputTxt+"[EOS]"
    inputTxt = inputTxt.lower() if p_forceLowerCase else inputTxt
    conversationHistoryTxt += inputTxt+'\n'
    inputTokens = textToTokens(inputTxt,tokenizer)
    numInputTokens += len(inputTokens[0])
    conversationHistoryTokens = torch.cat((conversationHistoryTokens,inputTokens),dim=1)
    # print(f'conv: {tokensToText(conversationHistoryTokens,tokenizer)}')
    
    # call the model with the full conversation history
    startTs = time.time() * 1000.0
    with torch.no_grad():
        outputTokens = generateTextCached(
            model=model,
            idx=conversationHistoryTokens.to(device),
            maxNewTokens=p_tokensToGenerate,
            temperature=p_temperature,
            topK=p_topK,
            minP=p_minP,
            printNextToken=False,
            tokenizer=tokenizer)
    outputTokens = outputTokens.to('cpu')
        
    # remove the input tokens from the generated output sequence, then append the output to the conversation history
    slicedTokens = outputTokens[:,len(conversationHistoryTokens[0]):]
    numOutputTokens += len(slicedTokens[0])
    outputText = tokensToText(slicedTokens[:,:-1],tokenizer)
    conversationHistoryTxt+=outputText
    conversationHistoryTokens = torch.cat((conversationHistoryTokens,slicedTokens),dim=1)

    endTs = time.time() * 1000.0
    totalRunTimeMs += (endTs-startTs)

    # print(f"Runtime: {runtimeMs} ms. T/Ps: {(p_tokensToGenerate/(runtimeMs/1000.0))} [{outputTokens.shape[1]} tokens]")
    print(outputText)

conversationHistoryTxt='\n'
tokensPS = (numOutputTokens/(totalRunTimeMs/1000.0)) if numOutputTokens > 0 else 'n/a'
print(f'Done chatting! [{tokensPS} tps, {round(numInputTokens,2)} input tokens, {numOutputTokens} output tokens, {len(conversationHistoryTokens[0])} history tokens]')