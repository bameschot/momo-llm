import re
import json

WIKI_DATA_FOLDER = './dataset'
OUTPUT_DATA_FOLDER = './output'


def writeAppendFile(content, fileName):
    with open(OUTPUT_DATA_FOLDER+'/'+fileName,"a",encoding = "utf-8") as outputFile:
        outputFile.write(content+'\n')
        
def removeNonAsciiCharacters(text):
    return ''.join(i for i in text if ord(i)<128)

def readSimpleWikipedia(fileName):
    inputFilePath = WIKI_DATA_FOLDER+'/'+fileName
    print(f'Reading {inputFilePath}')

    titleSizeCutoff = 6
    articleSizeCutoff = 50
    totalOutputArticles = 0

    invalidTitleChars = '<|>|\||-|\"|\'|_|§| |\!|\–|\.\.\.|“|→|\,|\&'
    invalidLineChars = '<|>|\|'

    singleFileWriteName="wiki-simple-english-cleaned"

    with open(inputFilePath,"r",encoding = "utf-8") as input:    
        line = input.readline()
        article = ""
        previousTitle = ''
        title = ''
        while True:
            if line == '':
                break

            line = input.readline()
            line = removeNonAsciiCharacters(line)
            
            lineSplit = line.split(' ')

            isTitle = len(lineSplit) < titleSizeCutoff and line != '\n' and not re.match(invalidTitleChars,line) 

            if isTitle:
                previousTitle = title
                title = line[:-1]

                if(len(article.split(' ')) > articleSizeCutoff):
                    try:
                        totalOutputArticles += 1

                        article = article.replace('\n','').replace('()','').replace('( )','').replace('(-)','')

                        singleArticleWriteTitle = previousTitle.replace(' ','_')
                        print(f'writing: {previousTitle}')

                        # writeAppendFile(singleArticleWriteTitle,'titles.txt')
                        # writeAppendFile(article,singleArticleWriteTitle+'.txt')
                        writeAppendFile(article,singleFileWriteName+'.txt')
                        
                    except:
                        None

                article = f'++++++++++++++++{title}================\n'
                # article = f'\n{title}\n'
            
            elif line!='\n':
                if not any(line in s for s in invalidLineChars):
                    article += line
                else:
                    print(f'invalid: {line}')

    print(f'Done parsing, found {totalOutputArticles} count')

readSimpleWikipedia('wiki-simple-english-full.txt')