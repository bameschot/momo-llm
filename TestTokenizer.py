from GenerateText import textToTokens
from Tokenizers import initializeTokenizer


tokenizer = initializeTokenizer('sentencepiece','conversational-english-16k-lc-16000')

text = 'how are you today?[eos]'
tokens = textToTokens(text,tokenizer)

print(f'{text}\n{tokens}')
