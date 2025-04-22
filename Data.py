import os
import glob
import pickle
import gzip
from pathlib import Path

from Tokenizers import *

import sentencepiece

from torch.utils.data import Dataset, DataLoader
from torch import torch

TOKENIZER_INPUT_DATA_DIRECTORY = "./input-data"
TOKENIZER_PROCESSED_DATA_DIRECTORY = "./processed-data"


PROCESSED_TXT_OUTPUT_FILE_NAME = "./processed-text.txt"
PROCESSED_BIN_OUTPUT_FILE_NAME = "./processed-text.bin"

TOKENIZED_READ_BUFFER_SIZE = 1024*1024*5 #5mb


def readInputFilePaths(globPattern=TOKENIZER_INPUT_DATA_DIRECTORY+'/*.*'):
    print(f"Finding files for pattern: {globPattern}")
    patterns = globPattern.split("|")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True)) 
    return files

def readProcessedDataFile(directory=TOKENIZER_PROCESSED_DATA_DIRECTORY):
    with open(f'{directory}/{PROCESSED_TXT_OUTPUT_FILE_NAME}','r') as f:
        return f.read()

def tokenizeProcessedDataFile(tokenizer,directory=TOKENIZER_PROCESSED_DATA_DIRECTORY,bufferSize=-1):
    tokens = []
    with open(f'{directory}/{PROCESSED_TXT_OUTPUT_FILE_NAME}','r') as f:
        text = f.read(bufferSize)
        while(len(text)>0):
            tokens.extend(tokenizer.encode(text))
            text = f.read(bufferSize)
        return tokens

def trainSentencePieceTokenizer(
        inputFilePaths, 
        processedOutputFileDir=TOKENIZER_VOCABULARY_DIRECTORY,
        processedOutputFileName="vocab-training.txt",
        vocabSize=8000,
        vocabularyName="sp",
        newTrainingFile=True,
        forceLowerCase=False):
    foup = f"{processedOutputFileDir}/{processedOutputFileName}.txt"
    if not os.path.isfile(foup) or newTrainingFile:
        print("Starting a new tokenize training file")
        preprocessInputDataAsText(inputFilePaths,processedOutputFileDir,processedOutputFileName,True,forceLowerCase)

    fullModelName = f"{vocabularyName}-{vocabSize}"
    modelDir = f"{processedOutputFileDir}/{fullModelName}"
    Path(f"{modelDir}").mkdir(parents=True, exist_ok=True)

    sentencepiece.SentencePieceTrainer.train(input=f'{processedOutputFileDir}/{processedOutputFileName}.txt',
                                model_prefix=f'{modelDir}/{fullModelName}',
                                model_type="bpe",
                                vocab_size=vocabSize,
                                self_test_sample_size=0,
                                input_format="text",
                                character_coverage=0.9995,
                                num_threads=os.cpu_count()-2,
                                split_digits=True,
                                allow_whitespace_only_pieces=True,
                                byte_fallback=True,
                                train_extremely_large_corpus=True,
                                unk_surface=r" \342\201\207 ",
                                normalization_rule_name="identity",
                                max_sentence_length=300000)    

def processRawTextLine(line,forceLowerCase=False):
    line = line.strip().replace('\n\n','\n')
    if forceLowerCase:
        line = line.lower()

    return line

def preprocessInputDataAsText(inputFilePaths, processedOutputFileDir,processedOutputFileName,maintainLineBreaks=False,forceLowerCase=False): 
    print(f'Creating a vocabulary for files: {inputFilePaths}')
    Path(f"{processedOutputFileDir}").mkdir(parents=True, exist_ok=True)

    rawVocabulary = set()
    with open(f'{processedOutputFileDir}/{processedOutputFileName}.txt','w+',encoding = "utf-8") as joinedOutput:
        fileIdx = 0
        for inputFilePath in inputFilePaths: 
            if os.path.isdir(inputFilePath):
                continue

            print(f'Reading {inputFilePath}')
            with open(inputFilePath,"r",encoding = "utf-8") as input:
                text = ""
                
                #read the whole file in one go to avoid ugly midsentence/word breaks
                while True: 
                    line = input.readline()
                    if line == '':
                        break
                    text += " "
                    text += processRawTextLine(line,forceLowerCase)
                    if maintainLineBreaks:
                        text+='\n'

                print(f"{inputFilePath} text size: {len(text)}")
                
                #split text into tokens
                for token in splitAndStripTextIntoTokens(text):
                    rawVocabulary.add(token)

                #write the text to the joined output with the end of text technical token
                if fileIdx > 0:
                    joinedOutput.write(f'\n{END_OF_TEXT_VOCAB_WORD}\n')
                joinedOutput.write(text.strip())
                fileIdx+=1

    #Add the technical tokens to the end vocabulary
    rawVocabulary.add(UKNOWN_VOCAB_WORD)
    rawVocabulary.add(END_OF_TEXT_VOCAB_WORD)
    vocabulary = {token:integer for integer,token in enumerate(sorted(rawVocabulary))}
    return vocabulary

def preprocessInputDataAsTokens(inputFilePaths, processedOutputFileDir,processedOutputFileName,tokenizer,partSizeMb=500,forceLowerCase=False): 
    print(f'Creating a binary tokenizer for files: {inputFilePaths}')
    Path(f"{processedOutputFileDir}/{processedOutputFileName}").mkdir(parents=True, exist_ok=True)


    tokenListCutoff = partSizeMb*1048576/4 #int
    fileIdx = 0
    outputFileIndex = 0
    tokens = []
    for inputFilePath in inputFilePaths: 
        if os.path.isdir(inputFilePath):
            continue

        print(f'Reading {inputFilePath}')
        with open(inputFilePath,"r",encoding = "utf-8") as input:
            text = ""
            
            #read the whole file in one go to avoid ugly midsentence/word breaks
            while True: 
                line = input.readline()
                if line == '':
                    break
                text += " "
                text += processRawTextLine(line,forceLowerCase)

            
            #split text into tokens
            tokens.extend(tokenizer.encode(text))
            print(f"{inputFilePath} tokensSize: {len(tokens)} / {tokenListCutoff}")
            del text

            #write the text to the joined output with the end of text technical token
            #if fileIdx > 0:
            #    joinedOutput.write(END_OF_TEXT_VOCAB_WORD)
            
            fileIdx+=1

        #if the token list exceeds the cutoff point write it to a batch file
        if(len(tokens) > tokenListCutoff):
            outputPath = f'{processedOutputFileDir}/{processedOutputFileName}/{processedOutputFileName}-{outputFileIndex}.bin'
            picleTokenListToGzipBin(tokens=tokens,filePath=outputPath)
            print(f"Writing processed output file: {outputPath}")
            outputFileIndex = outputFileIndex+1
            tokens.clear()
    
    #write the remainder of the tokens to the file
    outputPath = f'{processedOutputFileDir}/{processedOutputFileName}/{processedOutputFileName}-{outputFileIndex}.bin'
    picleTokenListToGzipBin(tokens=tokens,filePath=outputPath)
    print(f"Writing final processed output file: {outputPath}")
    
    vocabulary = {}
    return vocabulary

def picleTokenListToGzipBin(tokens,filePath):
    with gzip.open(filePath,'wb+') as joinedCompressedOutput:
        #Add the technical tokens to the end vocabulary
        print(f"Writing pickle for {len(tokens)} to {filePath}")
        pickle.dump(tokens,joinedCompressedOutput)

class GPTDatasetV2(Dataset):
    def __init__(self,text, tokenizer, maxLength, stride):
        self.tokenizer = tokenizer
        self.sourceIds = []
        self.targetIds = []

        #The entire text tokenized
        tokenIds = tokenizer.encode(text)

        #For the entire tokenized text provide the source and target blocks offset by the provided stride
        for idx in range(0, len(tokenIds)-maxLength,stride):
            src = tokenIds[idx:idx+maxLength]
            tar = tokenIds[idx+stride:idx+maxLength+stride]
            self.sourceIds.append(torch.tensor(src))
            self.targetIds.append(torch.tensor(tar))
    
    def __len__(self):
        return len(self.sourceIds)
    
    def __getitem__(self,idx):
        return self.sourceIds[idx], self.targetIds[idx]


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class GPTTokenizedDatasetV1(Dataset):
    def __init__(self, tokens, vocabSize, max_length, stride,device):
        self.input_ids = []
        self.target_ids = []

        #dataType = torch.int32 #torch.int32 if vocabSize <= 32767 else torch.int32

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1: i + max_length + 1]
            if(device != None):
                self.input_ids.append(torch.tensor(input_chunk).to(device))
                self.target_ids.append(torch.tensor(target_chunk).to(device))
            else:
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
                
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_text_dataloader_v1(txt,tokenizer ,batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def create_tokenized_dataloader_v1(tokens,tokenizer ,batch_size, max_length, stride,device,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = GPTTokenizedDatasetV1(tokens,tokenizer.vocabSize(), max_length, stride,device)
 
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


def createDataLoaderV1(tokenizer, text,numWorkers=0, batchSize=4,maxLength=256,stride=128,shuffle=True,dropLast=True):
    dataSet = GPTDatasetV2(tokenizer=tokenizer, text=text,maxLength=maxLength,stride=stride)
    return DataLoader(
        dataset=dataSet,
        batch_size=batchSize,
        shuffle=shuffle,
        drop_last=dropLast,
        num_workers=numWorkers
    )
