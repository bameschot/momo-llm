
import json

SQUAD_DATA_FOLDER = './dataset'
OUTPUT_DATA_FOLDER = './transformed'


def writeAppendFile(content, fileName):
    with open(OUTPUT_DATA_FOLDER+'/'+fileName,"a",encoding = "utf-8") as outputFile:
        outputFile.write(content+'\n')
        


def readSquadDataAsJsonObject(fileName):
    inputFilePath = SQUAD_DATA_FOLDER+'/'+fileName
    print(f'Reading {inputFilePath}')
    suadJson = None
    with open(inputFilePath,"r",encoding = "utf-8") as input:    
        suadJson = json.load(input)
    
    for dataItem in suadJson["data"]:
        title = dataItem["title"]
        print(title)
        # print(context)
        fullContext = ""
        for paragraph in dataItem["paragraphs"]:
            context = paragraph["context"]
            fullContext += '\n' + context
            for qasItem in paragraph["qas"]:
                question = qasItem["question"]
                previousAnswers = []
                for answerItem in qasItem["answers"]:
                    answer = answerItem["text"]
                    if(answer not in previousAnswers):
                        print(question + " = "+ answer)
                        previousAnswers.append(answer)
        
        writeAppendFile(fullContext,title)
    





readSquadDataAsJsonObject('train-v2.0.json')