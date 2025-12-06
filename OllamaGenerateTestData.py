import argparse
import time 

from ollama import chat
from ollama import ChatResponse
from pathlib import Path

from DataPreProcessing import loadReplacementFile, processRawTextLineWithReplacements



########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Generate test data using a locally runnong ollama innstance"
    )
parser.add_argument("--model", type=str,default='llama3.2:3b', help="The model that the data is generated on")
parser.add_argument("--system", type=str,default="you are a conversation generator, you generate a conversation between 2 participants and ensure you just output the conversation line by line and no other text", help="The system prompt provided to the model")
parser.add_argument("--prompt", type=str,default="generate a casual conversation between 300 and 2000 words, only output the conversations and no other text, put the participants names between brackets [$name], do not use names in the conversation", help="The prompt provided to the model when generating data")
parser.add_argument("--fileSplitSizeMb", type=int,default=50, help="the desired size of the output file before being split")
parser.add_argument("--outputFileFolder", type=str,default='./input-data', help="the root folder that the output files will be written to")
parser.add_argument("--outputFileName", type=str,default='syntetic-conversations', help="the name of the output files")
parser.add_argument("--replacementFilePath", type=str,default='./_ollama-generation-replacements.rep', help="the path of the replacement file used to format the models output")
parser.add_argument("--printInterval", type=int,default=10, help="the interval that a control is being printed")
parser.add_argument("--numberOfGenerations", type=int,default=50, help="the number of responses that are generated using the prompt from the model")

parser.add_argument("--newTrainingFile", action='store_true',help="Indicates if the output file must be new or if the data has to be appended")

args = parser.parse_args()

p_model = args.model
p_system = args.system
p_prompt = args.prompt
p_fileSplitSizeMb = args.fileSplitSizeMb
p_outputFileFolder = args.outputFileFolder
p_outputFileName = args.outputFileName
p_replacementFilePath = args.replacementFilePath
p_printInterval= args.printInterval
p_newTrainingFile = args.newTrainingFile
p_numberOfGenerations = args.numberOfGenerations


#model = 'tinyllama:1.1b'
#model = 'ministral-3:3b'
#model = 'llama3.2:3b'
#system = 'you are a conversation generator, you generate a conversation between 2 participants and ensure you just output the conversation line by line and no other text'
#content = 'generate a casual conversation between 300 and 2000 words, only output the conversations and no other text, put the participants names between brackets [$name], do not use names in the conversation'
#fileSplitSizeMb = 50
#numberOfGenerations = 100
#outputFileFolder = './input-data'
#outputFileName = 'syntetic-conversations'
#replacementFilePath = './_ollama-generation-replacements.rep'
#printInterval = 10

########################################
#Functions
########################################

def writeTextToFile(outputFolder,outputFileName,outputFileIdx, content,newFile=True):
    outputPath = f'{outputFolder}/{outputFileName}/{outputFileName}-{outputFileIdx:03d}.txt'
    Path(f"{outputFolder}/{outputFileName}").mkdir(parents=True, exist_ok=True)
    writeMode = 'w' if newFile else 'a'
    with open(outputPath,writeMode,encoding = "utf-8") as output:
        print(f'writing output file ({writeMode}): {output.name}')
        output.write(content)

########################################
#Script
########################################

text = ""
fileIdx = 0
charactersWritten = 0
replacementDefinition = loadReplacementFile(p_replacementFilePath)
interationStartTimeS = time.time()
generationStartTimeS = interationStartTimeS

print(f'Start generating data with\n- model: {p_model}\n-system prompt {p_system}\n-prompt:{p_prompt}')
for genIdx in range(p_numberOfGenerations):  
    response: ChatResponse = chat(model=p_model, messages=[
    {
        'system': p_system,
        'role': 'user',
        'content': p_prompt,
    },
    ])
    #print(response['message']['content'])
    # or access fields directly from the response object
    messageContent = response.message.content+'\n'
    processedContent = processRawTextLineWithReplacements(messageContent,False,replacementDefinition)
    charactersWritten += len(processedContent)
    text += processedContent + "=====================\n"

    if genIdx > 0 and genIdx % p_printInterval == 0:
        print(processedContent)
        print(f'----------------------------------------------------------------------------------------')
        print(f'generated {genIdx:9d} responses with {charactersWritten} written characters [{(time.time()-interationStartTimeS)/p_printInterval} seconds]')
        interationStartTimeS = time.time()
        writeTextToFile(p_outputFileFolder,p_outputFileName,fileIdx,text,False)
        text = ""
        if(len(text)>p_fileSplitSizeMb*1024*1024):
            fileIdx+=1
            writeTextToFile(p_outputFileFolder,p_outputFileName,fileIdx,text,p_newTrainingFile)


writeTextToFile(p_outputFileFolder,p_outputFileName,fileIdx,text,p_newTrainingFile)
print('Done generating data')
print(f'generated {genIdx:9d} responses with {charactersWritten} written characters [{(time.time()-generationStartTimeS)/p_numberOfGenerations} seconds per request] in {(time.time()-generationStartTimeS)} seconds')
