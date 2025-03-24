import argparse

from Data import *

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Train a sentencepiece tokenizer for the request vocabulary size"
    )
parser.add_argument("--inputDirectory", type=str,default=TOKENIZER_INPUT_DATA_DIRECTORY, help="The base directory from where to load the data")
parser.add_argument("--globPattern", type=str,default="**/**", help="The glob pattern that determines the file to load data fromfrom the base directory")
parser.add_argument("--outputDirectory", type=str,default=TOKENIZER_VOCABULARY_DIRECTORY, help="The base directory to write the output to")
parser.add_argument("--vocabSize", type=int,default=4096, help="The desired size of the tokenizer vocabulary")
parser.add_argument("--vocabularyName", type=str,default="tkn", help="Name of the sentencepiece vocabulary")

parser.add_argument("--newTrainingFile", action='store_true',help="Indicates if the output file must be new or if the data has to be appended")

args = parser.parse_args()

p_inputDirectory = args.inputDirectory
p_globPattern = args.globPattern
p_outputDirectory = args.outputDirectory
p_vocabSize = args.vocabSize
p_vocabularyName = args.vocabularyName
p_newTrainingFile= args.newTrainingFile


print(f"start training tokenizer for vocab size {p_vocabSize}: {p_inputDirectory}/{p_globPattern}")
trainSentencePieceTokenizer(readInputFilePaths(p_inputDirectory,p_globPattern),p_outputDirectory,"sp-training",p_vocabSize,p_vocabularyName,p_newTrainingFile)

print(f"Done training tokenizer for vocab size {p_vocabSize}")
