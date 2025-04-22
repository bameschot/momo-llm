import argparse

from Data import *

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Train a sentencepiece tokenizer for the request vocabulary size"
    )
parser.add_argument("--inputData", type=str,default=TOKENIZER_PROCESSED_DATA_DIRECTORY, help="The glob pattern for selecting input data")
parser.add_argument("--globPattern", type=str,default="**/**", help="The glob pattern that determines the file to load data fromfrom the base directory")
parser.add_argument("--outputDirectory", type=str,default=TOKENIZER_VOCABULARY_DIRECTORY, help="The base directory to write the output to")
parser.add_argument("--vocabSize", type=int,default=4096, help="The desired size of the tokenizer vocabulary")
parser.add_argument("--vocabularyName", type=str,default="tkn", help="Name of the sentencepiece vocabulary")
parser.add_argument("--forceLowerCase", action='store_true', help="ensures that all text becomes lowercase")

parser.add_argument("--newTrainingFile", action='store_true',help="Indicates if the output file must be new or if the data has to be appended")

args = parser.parse_args()

p_inputData = args.inputData
p_globPattern = args.globPattern
p_outputDirectory = args.outputDirectory
p_vocabSize = args.vocabSize
p_vocabularyName = args.vocabularyName
p_newTrainingFile = args.newTrainingFile
p_forceLowerCase = args.forceLowerCase


print(f"start training tokenizer for vocab size {p_vocabSize}: {p_inputData}")
forceLowerCaps = p_forceLowerCase if p_forceLowerCase else False
trainSentencePieceTokenizer(readInputFilePaths(p_inputData),p_outputDirectory,"sp-training",p_vocabSize,p_vocabularyName,p_newTrainingFile,forceLowerCaps)

print(f"Done training tokenizer for vocab size {p_vocabSize}")
