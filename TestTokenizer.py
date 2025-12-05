from GenerateText import textToTokens
from Tokenizers import initializeTokenizer


tokenizer = initializeTokenizer('sentencepiece','synth-english-10k-lc-10000')

text = 'how are you today?[eos]'
tokens = textToTokens(text,tokenizer)

print(f'{text}\n{tokens}')
