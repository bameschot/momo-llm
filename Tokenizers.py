import re

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
    if tokenizerType == 'gpt2':
        return GPT2Tokenizer()
    elif tokenizerType == 'sentencepiece':
        return SentencePieceTokenizer(f"{TOKENIZER_VOCABULARY_DIRECTORY}/{tokenizerName}/{tokenizerName}.model")
    elif tokenizerType == 'simple':
        return SimpleTokenizerV2(f"{TOKENIZER_VOCABULARY_DIRECTORY}/{tokenizerName}/{tokenizerName}.vocab")
    else:
        return GPT2Tokenizer()

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
    
class SentencePieceTokenizer:
    def __init__(self,modelFile):
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=modelFile)

    def encode(self,text):
        return self.tokenizer.encode(text)
    
    def decode(self,ids):
        return self.tokenizer.decode(ids)
    
    def vocabSize(self):
        return self.tokenizer.vocab_size()
    
