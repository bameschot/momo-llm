import re
import os
import tiktoken
from  torch.utils.data import Dataset, DataLoader
from torch import torch

TOKENIZER_INPUT_DATA_DIRECTORY = "./input-data"
TOKENIZER_PROCESSED_DATA_DIRECTORY = "./processed-data"


PROCESSED_TXT_OUTPUT_FILE_NAME = "./processed-text.txt"
TOKENIZE_INTERPUNCTION_REGEX= r'\s+([,.:;?_!"()\'])'
TOKENIZE_REGEX = r'([,.:;?_!"()\']|--|\s)'
TOKENIZED_READ_BUFFER_SIZE = 1024*1024*5 #5mb

UKNOWN_VOCAB_WORD = "<|unk|>"
END_OF_TEXT_VOCAB_WORD = "<|endoftext|>"

def readInputFilePaths(directory=TOKENIZER_INPUT_DATA_DIRECTORY):
    return [f'{directory}/{fp}' for fp in os.listdir(directory)]

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

def preprocessInputData(inputFilePaths, processedOutputFileDir): 
    print(f'Creating a vocabulary for files: {inputFilePaths}')

    rawVocabulary = set()
    with open(f'{processedOutputFileDir}/{PROCESSED_TXT_OUTPUT_FILE_NAME}','w+',encoding = "utf-8") as joinedOutput:
        fileIdx = 0
        for inputFilePath in inputFilePaths: 
            print(f'Reading {inputFilePath}')
            with open(inputFilePath,"r",encoding = "utf-8") as input:
                #read the whole file in one go to avoid ugly midsentence/word breaks
                text = input.read()
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

class GPTDatasetV1(Dataset):
    def __init__(self,directory, tokenizer, maxLength, stride):
        self.tokenizer = tokenizer
        self.sourceIds = []
        self.targetIds = []

        #The entire text tokenized
        tokenIds = tokenizer.encode(readProcessedDataFile(directory))

        #For the entire tokenized text provide the source and target blocks offset by the provided stride
        for idx in range(0, len(tokenIds)-maxLength,stride):
            src = tokenIds[idx:idx+maxLength]
            tar = tokenIds[idx+stride:idx+maxLength+stride]
            self.sourceIds.append(torch.tensor(src))
            self.targetIds.append(torch.tensor(tar))
        
        del tokenIds

    def __len__(self):
        return len(self.sourceIds)
    
    def __getItem__(self,idx):
        return self.sourceIds[idx], self.targetIds[idx]

class GPTDatasetV2(Dataset):
    def __init__(self,directory, tokenizer, maxLength, stride, readBufferSize=1024*1024*4):
        self.tokenizer = tokenizer
        self.sourceIds = []
        self.targetIds = []

        #The entire text tokenized
        tokenIds = tokenizeProcessedDataFile(directory=directory,tokenizer=tokenizer,bufferSize=readBufferSize)

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

def createDataLoaderV1(tokenizer, directory=TOKENIZER_PROCESSED_DATA_DIRECTORY,numWorkers=0, batchSize=4,maxLength=256,stride=128,shuffle=True,dropLast=True):
    dataSet = GPTDatasetV2(tokenizer=tokenizer, directory=directory,maxLength=maxLength,stride=stride)
    return DataLoader(
        dataset=dataSet,
        batch_size=batchSize,
        shuffle=shuffle,
        drop_last=dropLast,
        num_workers=numWorkers
    )
    
class embedd

vocab = preprocessInputData(readInputFilePaths(TOKENIZER_INPUT_DATA_DIRECTORY),TOKENIZER_PROCESSED_DATA_DIRECTORY)
gptTokenizer = GPT2Tokenizer()
print(f'vocabulary size: {len(vocab)} tokenizer size: {gptTokenizer.vocabSize()}') 


#dataloader of lenghh and stride 4
maxLength = 4
dataloader = createDataLoaderV1(tokenizer=gptTokenizer,batchSize=8,maxLength=maxLength,stride=maxLength,shuffle=False)

#create embedding layer of the vocabulary size and the desired output dimension
vocabSize = gptTokenizer.vocabSize()
outputDimension = 256 #configurable, 12228 for gpt3
embeddingLayer = torch.nn.Embedding(vocabSize, outputDimension)

#create a positional embedding layer
positionalEmbeddingLayer = torch.nn.Embedding(vocabSize,outputDimension)
positionalEmbedding = positionalEmbeddingLayer(torch.arange(maxLength))


#apply the embedding to a single batch of data from the dataloader
dataIter = iter(dataloader)
input = next(dataIter)[0]
print(f"input tensor shape {input.shape}")
embeddedInputs = embeddingLayer(input) + positionalEmbedding
print(f"embedding shape {embeddingLayer.__sizeof__()}")

