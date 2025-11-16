import argparse

from Data import *
from GPTModelConfig import *

########################################
#Functions
#######################################
def splitTextInputFiles(inputFilePaths, outputDir,processedOutputFileName,partSizeMb=500): 
    print(f'Splitting files tokenizer for: {inputFilePaths}')
    Path(f"{outputDir}/{processedOutputFileName}").mkdir(parents=True, exist_ok=True)


    tokenListCutoff = partSizeMb*1048576 #int
    fileIdx = 0
    outputFileIndex = 0
    outputText = ""
    for inputFilePath in inputFilePaths: 
        if os.path.isdir(inputFilePath):
            continue

        print(f'Reading {inputFilePath}')
        with open(inputFilePath,"r",encoding = "utf-8") as input:
            
            #read the whole file and tokenize line by line
            lineCount = 0
            while True: 
                lineCount+=1
                line = input.readline()
                if line == '':
                    break
                outputText += line

                if lineCount % 100000 == 0:
                    print(f'{inputFilePath}: read {lineCount} lines')

                #if the token list exceeds the cutoff point write it to a batch file
                if(len(outputText) > tokenListCutoff):
                    outputPath = f'{outputDir}/{processedOutputFileName}/{processedOutputFileName}-{outputFileIndex:05d}.txt'
                    with open(outputPath,'w') as splitOutput:
                        #Add the technical tokens to the end vocabulary
                        print(f"Writing split output for {len(outputText)} to {outputPath}")
                        splitOutput.write(outputText)
                    outputFileIndex = outputFileIndex+1
                    outputText = ""
                    gc.collect()
            
            fileIdx+=1
    
    #write the remainder of the tokens to the file
    outputPath = f'{outputDir}/{processedOutputFileName}/{processedOutputFileName}-{outputFileIndex:05d}.txt'
    with open(outputPath,'w') as splitOutput:
        #Add the technical tokens to the end vocabulary
        print(f"Writing split output for {len(outputText)} to {outputPath}")
        splitOutput.write(outputText)
        outputFileIndex = outputFileIndex+1
        outputText = ""
    
    gc.collect()
    print(f"Writing final split output file: {outputPath}")

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Splitting Text Data files"
    )
parser.add_argument("--inputData", type=str,default=TOKENIZER_INPUT_DATA_DIRECTORY, help="The glob pattern for selecting input data")

parser.add_argument("--outputDirectory", type=str,default=TOKENIZER_INPUT_DATA_DIRECTORY, help="The base directory to write the split output to")
parser.add_argument("--outputFileName", type=str,default="processed-text-full-wiki", help="the file name of the processed file")
parser.add_argument("--outputBatchSizeMb", type=int,default=500, help="The (inflated) batch size of the output file in megabytes, the input file(s) will be aggregated to the indicated size size and dumped as <file-name>-<idx>.(txt|bin)")

args = parser.parse_args()

p_inputData = args.inputData

p_outputDirectory = args.outputDirectory
p_outputFileName = args.outputFileName
p_outputBatchSizeMb = args.outputBatchSizeMb


########################################
#Script
########################################


print(f"start splitting: {p_inputData}")

splitTextInputFiles(readInputFilePaths(p_inputData),p_outputDirectory,p_outputFileName,p_outputBatchSizeMb)

print("Done splitting")


