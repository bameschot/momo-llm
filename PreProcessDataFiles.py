import argparse

from Data import *
from GPTModelConfig import *

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Preprocesses GPT Data files"
    )
parser.add_argument("--inputData", type=str,default=TOKENIZER_PROCESSED_DATA_DIRECTORY, help="The glob pattern for selecting input data")
parser.add_argument("--modelConfig", type=str,default="GPT_CONFIG_SMALL_CTX512_8_8_512", help="Determines the model configuration that a new model is initialised with, this parameter is ignored for models loaded from a checkpoint")
parser.add_argument("--outputDirectory", type=str,default=TOKENIZER_PROCESSED_DATA_DIRECTORY, help="The base directory to write the output from")
parser.add_argument("--outputFileName", type=str,default="processed-text-full-wiki", help="the file name of the processed file")
parser.add_argument("--outputBatchSizeMb", type=int,default=500, help="The (inflated) batch size of the output file in megabytes, the input file(s) will be aggregated to the indicated size size and dumped as <file-name>-<idx>.(txt|bin)")
parser.add_argument("--forceLowerCase", action='store_true', help="ensures that all text becomes lowercase")

parser.add_argument("--newOutputFile", action='store_true',help="Indicates if the output file must be new or if the data has to be appended")
parser.add_argument("--isTokenized", action='store_true',help="Indicates if the output file is tokenized with the gpt2 tokenizer")

args = parser.parse_args()

p_inputData = args.inputData
p_outputDirectory = args.outputDirectory
p_outputFileName = args.outputFileName
p_outputBatchSizeMb = args.outputBatchSizeMb
p_modelConfig=args.modelConfig
p_forceLowerCase= args.forceLowerCase

p_newFile = args.newOutputFile
p_isTokenized = args.isTokenized



print(f"start preprocessing: {p_inputData}")
forceLowerCase = p_forceLowerCase if p_forceLowerCase else False

if p_isTokenized:
    trainingConfig = modelConfigs[p_modelConfig]
    tokenizer = initializeTokenizer(trainingConfig[TOKENIZER_TYPE],trainingConfig[TOKENIZER_NAME])
    preprocessInputDataAsTokens(readInputFilePaths(p_inputData),p_outputDirectory,p_outputFileName,tokenizer,p_outputBatchSizeMb,forceLowerCase)
else:
    vocab = preprocessInputDataAsText(readInputFilePaths(p_inputData),p_outputDirectory,p_outputFileName,False,forceLowerCase)
    print(f"vocab size {len(vocab)}")

print("Done preprocessing")
