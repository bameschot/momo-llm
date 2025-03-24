import argparse

from Data import *

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Preprocesses GPT Data files"
    )
parser.add_argument("--inputDirectory", type=str,default=TOKENIZER_INPUT_DATA_DIRECTORY, help="The base directory from where to load the data")
parser.add_argument("--globPattern", type=str,default="**/**", help="The glob pattern that determines the file to load data fromfrom the base directory")
parser.add_argument("--outputDirectory", type=str,default=TOKENIZER_PROCESSED_DATA_DIRECTORY, help="The base directory to write the output from")
parser.add_argument("--outputFileName", type=str,default="processed-text-full-wiki", help="the file name of the processed file")
parser.add_argument("--outputBatchSizeMb", type=int,default=500, help="The (inflated) batch size of the output file in megabytes, the input file(s) will be aggregated to the indicated size size and dumped as <file-name>-<idx>.(txt|bin)")

parser.add_argument("--newOutputFile", action='store_true',help="Indicates if the output file must be new or if the data has to be appended")
parser.add_argument("--isTokenized", action='store_true',help="Indicates if the output file is tokenized with the gpt2 tokenizer")

args = parser.parse_args()

p_inputDirectory = args.inputDirectory
p_globPattern = args.globPattern
p_outputDirectory = args.outputDirectory
p_outputFileName = args.outputFileName
p_outputBatchSizeMb = args.outputBatchSizeMb

p_newFile = args.newOutputFile
p_isTokenized = args.isTokenized



print(f"start preprocessing: {p_inputDirectory}/{p_globPattern}")
if p_isTokenized:
    preprocessInputDataAsTokens(readInputFilePaths(p_inputDirectory,p_globPattern),p_outputDirectory,p_outputFileName,GPT2Tokenizer(),p_outputBatchSizeMb)
else:
    vocab = preprocessInputDataAsText(readInputFilePaths(p_inputDirectory,p_globPattern),p_outputDirectory,p_outputFileName)
    print(f"vocab size {len(vocab)}")

print("Done preprocessing")
