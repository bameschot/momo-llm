import re
import os
import tiktoken
import glob
import pickle
import gzip

from pathlib import Path

from  torch.utils.data import Dataset, DataLoader
from torch import torch

TOKENIZER_INPUT_DATA_DIRECTORY = "./input-data"
TOKENIZER_PROCESSED_DATA_DIRECTORY = "./processed-data"


PROCESSED_TXT_OUTPUT_FILE_NAME = "./processed-text.txt"
PROCESSED_BIN_OUTPUT_FILE_NAME = "./processed-text.bin"

TOKENIZE_INTERPUNCTION_REGEX= r'\s+([,.:;?_!"()\'])'
TOKENIZE_REGEX = r'([,.:;?_!"()\']|--|\s)'
TOKENIZED_READ_BUFFER_SIZE = 1024*1024*5 #5mb

UKNOWN_VOCAB_WORD = "<|unk|>"
END_OF_TEXT_VOCAB_WORD = "<|endoftext|>"

def readInputFilePaths(directory=TOKENIZER_INPUT_DATA_DIRECTORY,globPattern='/**/*.*'):
    files = glob.glob(directory + globPattern, recursive=True)
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

def preprocessInputDataAsText(inputFilePaths, processedOutputFileDir,processedOutputFileName): 
    print(f'Creating a vocabulary for files: {inputFilePaths}')

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
                    text += line.strip()

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

def preprocessInputDataAsTokens(inputFilePaths, processedOutputFileDir,processedOutputFileName,tokenizer,partSizeMb=500): 
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
                text += line.strip()

            
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

def splitAndStripTextIntoTokens(text):
    #split on the token regex
    rawTokens = re.split(TOKENIZE_REGEX,text)
    return [rawToken.strip() for rawToken in rawTokens if rawToken.strip()]
                    
class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.word_to_id = vocab
        self.id_to_word = {i:s for s,i in vocab.items()}

    def encode(self,text):
        return [ self.word_to_id[w]if w in self.word_to_id else self.word_to_id[UKNOWN_VOCAB_WORD] for w in splitAndStripTextIntoTokens(text)]
    
    def decode(self,ids):
        return re.sub(TOKENIZE_INTERPUNCTION_REGEX,r'\1'," ".join([self.id_to_word[i] for i in ids]))
    
    def vocabSize(self):
        return len(self.word_to_id)

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self,text):
        return self.tokenizer.encode(text=text,allowed_special={END_OF_TEXT_VOCAB_WORD})
    
    def decode(self,ids):
        return self.tokenizer.decode(ids)
    
    def vocabSize(self):
        return 50257 #hardcoded for tiktoken gpt2


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
    def __init__(self, tokens, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokens

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


def create_text_dataloader_v1(txt,tokenizer ,batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def create_tokenized_dataloader_v1(tokens,tokenizer ,batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = GPTTokenizedDatasetV1(tokens, max_length, stride)

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
    