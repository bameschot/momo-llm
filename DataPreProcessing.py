import os
import re
import glob


SPLIT_REPLACEMENT_DEFINITION = '-->'


def loadReplacementDefinitions(rootFolder):
    replacementFiles = loadReplacementFiles(rootFolder)
    print(f'loading replacement files: {replacementFiles}')
    replacementDefinitions = {}
    for replacementFilename in replacementFiles:
        replacementDefinitions[replacementFilename[:replacementFilename.rfind('/')]] = loadReplacementFile(replacementFilename)
    
    return replacementDefinitions

def findReplacementDefinition(dataFileName, replacementDefinitions:dict):
    for key in sorted(replacementDefinitions.keys(), key=len, reverse=True):
        if key in dataFileName:
            return replacementDefinitions[key]

def loadReplacementFile(fileName:str):
    if not os.path.exists(fileName):
        return list()
    
    with open(f'{fileName}','r') as f:
        line = f.readline()
        replacements = list()
        while(len(line)>0):
            if line[0] != '#':  
                split = line[:-1].split(SPLIT_REPLACEMENT_DEFINITION)
                source = split[0]
                target = '' if len(split) == 1 else split[1]

                print(f'loading replacement for {fileName}: {source}-->{target}')
                replacements.append((source,target))
            else: 
                print(f'commented replacement for {fileName}: {line[:-1]}')

            line = f.readline()

    return replacements

def loadReplacementFiles(rootFolder):
    return glob.glob(rootFolder+'/**/**.rep', recursive=True)


def processRawTextLineWithReplacements(line,forceLowerCase=False,replacements:list=list()):
    if replacements:
        for replacement in replacements:
            # line = line.replace(replacement[0],replacement[1])
            line = re.sub(replacement[0],replacement[1],line)
            # print(f'{line} == replaced {replacement[0]} with {replacement[1]}')
        
    if forceLowerCase:
        line = line.lower()

    return line

