import os
from pathlib import Path
import re
import uuid

import tiktoken
import sentencepiece

TOKENIZER_VOCABULARY_DIRECTORY = "./vocabulary"

TOKENIZE_INTERPUNCTION_REGEX= r'\s+([,.:;?_!"()\'])'
TOKENIZE_REGEX = r'([,.:;?_!"()\']|--|\s)'
UKNOWN_VOCAB_WORD = "<|unk|>"
END_OF_TEXT_VOCAB_WORD = "<|endoftext|>"



def splitAndStripTextIntoTokens(text):
    #split on the token regex
    rawTokens = re.split(TOKENIZE_REGEX,text)
    return [rawToken.strip() for rawToken in rawTokens if rawToken.strip()]

def initializeTokenizer(tokenizerType,tokenizerName):
    if tokenizerType == 'sentencepiece':
        return SentencePieceTokenizer(modelFileName = tokenizerName)

def initializeTokenizerFromModelBytes(tokenizerType,modelName,modelFileBytes):
    if tokenizerType == 'sentencepiece':
        return SentencePieceTokenizer(modelFileName = modelName, modelFileBytes = modelFileBytes)



class SentencePieceTokenizer:
    def __init__(self,modelFileName, modelFileBytes=None):
        modelFileFolder = f"{TOKENIZER_VOCABULARY_DIRECTORY}/{modelFileName}" 
        
        if modelFileBytes is None:
            modelFilePath = f"{modelFileFolder}/{modelFileName}.model" 
            print(f'Initialize tokenizer from external model file: {modelFilePath}')
            with open(modelFilePath,'rb') as mf:
                self.modelFileBytes = mf.read(-1)
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=modelFilePath)
        else:
            modelFileFolder = f"{modelFileFolder}-{uuid.uuid4().hex}"
            modelFilePath = f"{modelFileFolder}/{modelFileName}.model" 

            Path(modelFileFolder).mkdir(parents=True, exist_ok=True)
            print(f'Initialize tokenizer from internal model file: {modelFilePath}')
            self.modelFileBytes = modelFileBytes
            with open(modelFilePath,'wb') as mf:
                mf.write(modelFileBytes)
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=modelFilePath)

            os.remove(modelFilePath)
            Path(modelFileFolder).rmdir()
                
        self.tokenizer.eos_id = 3
        #self.tokenizer.SetEncodeExtraOptions('eos')        

    def encode(self,text):
        return self.tokenizer.encode(text)
    
    def decode(self,ids):
        return self.tokenizer.decode(ids)
    
    def vocabSize(self):
        return self.tokenizer.vocab_size()

    
