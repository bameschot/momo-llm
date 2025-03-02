import time

import torch
import torch.nn as nn
import torch.optim

from Preprocessing import GPT2Tokenizer, createDataLoaderV1, create_dataloader_v1
from GenerateText import generateText, simpleTextGeneration
from GPTModel import GPTModel
from GPTModelConfig import *

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


MODEL_FOLDER = "./models"

def textToTokens(text,tkz):
    encoded = tkz.encode(text)
    return torch.tensor(encoded).unsqueeze(0)

def tokensToText(tokens,tkz):
    flat = tokens.squeeze(0)
    return tkz.decode(flat.tolist())


def calculationLossBatch(inputBatch,targetBatch,model,device):
    #move the batches to the desired device
    inputBatch = inputBatch.to(device)
    targetBatch = targetBatch.to(device)

    #call the model to generate the next output tokens
    logits = model(inputBatch)

    #calculate the loss using the cross entrophy 
    loss = nn.functional.cross_entropy(input=logits.flatten(0,1),target= targetBatch.flatten())
    return loss
    

def calculationLossLoader(dataLoader,model,device,numberOfBatches=None):
    totalLoss = 0
    #if no data is in the dataloader return nan
    if len(dataLoader)==0:
        return float("nan")
    #if the number of batches is not specified process all
    elif numberOfBatches is None:
        numberOfBatches = len(dataLoader)
    #set the number of batches to the requested number
    else:
        numberOfBatches = min(len(dataLoader),numberOfBatches)
    
    #iterate over the batches and calculate loss
    #add the loss to the total loss
    for i, (inputBatch,targetBatch) in enumerate(dataLoader):
        if i< numberOfBatches:
            loss = calculationLossBatch(inputBatch=inputBatch,targetBatch=targetBatch,model=model,device=device)
            totalLoss+=loss.item()
        else:
            break
    #return the average loss over all the batches
    return totalLoss/numberOfBatches

def evaluateModel(model,trainingDataLoader, validationDataLoader, device, evaluationIterations):
    #put the model in evaluation mode to prevent dropouts / etc from interfering with the results 
    model.eval()
    #calculate the loss results
    with torch.no_grad():
        trainingLoss = calculationLossLoader(trainingDataLoader,model,device,numberOfBatches=evaluationIterations)
        validationLoss = calculationLossLoader(validationDataLoader,model,device,numberOfBatches=evaluationIterations)
    #put the model back into training  mode
    model.train()
    return trainingLoss, validationLoss

def generateAndPrintSample(model, tokenizer, device, startContext):
    model.eval()
    encoded = textToTokens(startContext,tokenizer).to(device)
    with torch.no_grad():
        tokenIds = generateText(model,encoded,50)
        decodedText = tokensToText(tokenIds,tokenizer).replace("\n"," ")
        print(f"training text sample: {decodedText}")
    model.train()

def storeCheckPoint(modelName,model,optimizer):
    torch.save(
        {
            "ModelStateDict": model.state_dict(),
            "OptimizerStateDict": optimizer.state_dict()
        },
        f"{MODEL_FOLDER}/{modelName}.pth"
    )

def loadModel(modelName,config,device,learningRate=0.004,weightDecay=0.1):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}.pth",device)
    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
    optimizer.load_state_dict(modelData["OptimizerStateDict"])
    return model, optimizer


def trainModel(
        modelName,
        modelConfig,
        trainingDataLoader, 
        validationDataLoader, 
        device, 
        tokenizer,
        loadModelFromCheckpoint=True,
        learningRate=0.0004,
        weightDecay=0.1,
        numberOfEpochs=10, 
        evaluationFrequency=5, 
        evaluationIterations=5,
        checkpointStorageFrequency=10,
        startContext="Will you leave the green fields devoid capital"
        ):          
    if loadModelFromCheckpoint:
        model,optimizer = loadModel(modelName,modelConfig,device,learningRate,weightDecay)
        print(f"Loaded model {modelName} from file")
    else: 
        model = GPTModel(config).to(device)
        print(f"Starting new model {modelName}")
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
    model.train()
    print(f"Training model on {device}")
    
    return trainModelSimple(
        modelName=modelName,
        model=model,
        trainingDataLoader=trainingDataLoader,
        validationDataLoader=validationDataLoader,
        optimizer=optimizer,
        device=device,
        numberOfEpochs=numberOfEpochs,
        evaluationFrequency=evaluationFrequency,
        checkpointStorageFrequency=checkpointStorageFrequency,
        evaluationIterations=evaluationIterations,
        startContext=startContext,
        tokenizer=tokenizer
    )


def trainModelSimple(
        modelName,
        model, 
        trainingDataLoader, 
        validationDataLoader, 
        optimizer, 
        device, 
        tokenizer,
        numberOfEpochs, 
        evaluationFrequency, 
        evaluationIterations,
        checkpointStorageFrequency,
        startContext
        ):
    #set start data
    trainingLosses, validationLosses, trackTokensSeen = [],[],[]
    tokensSeen, globalStep = -1,0

    #iterate over the requested amount of epochs (complete batch runs)
    for epoch in range(numberOfEpochs):
        model.train(),
        startEpochTs = time.time() * 1000.0

        #iterate over the batches
        for inputBatch,targetBatch in trainingDataLoader:
            startTs = time.time() * 1000.0
            #reset the loss gradients
            optimizer.zero_grad()
            #calculate the loss gradients for the batch
            loss = calculationLossBatch(inputBatch,targetBatch,model,device)
            #execute the backwards pass
            loss.backward()
            #update the model weights with the loss gradients
            optimizer.step()
            tokensSeen += inputBatch.numel()
            globalStep +=1
            
            endTs = time.time() * 1000.0
            timePerStep=(endTs-startTs)

            #if the current step reaches the evaluation fequency
            if(globalStep%evaluationFrequency==0):
                #evaluate the model and get the training loss and validation loss
                trainingLoss, validationLoss = evaluateModel(model,trainingDataLoader,validationDataLoader, device, evaluationIterations)
                trainingLosses.append(trainingLoss)
                validationLosses.append(validationLoss)

                #print the current results
                trackTokensSeen.append(tokensSeen)
                print(f"Epoch: {epoch}; (Step {globalStep:06d}): Training Loss {trainingLoss:.3f} Validation Loss {validationLoss:.3f} avg. Step processing time {timePerStep} ms")

            #storage checkpoint interval
            if checkpointStorageFrequency is not None and globalStep%checkpointStorageFrequency==0:
                storeCheckPoint(modelName,model,optimizer)
                print(f"Storing model: {modelName}")
    
        #end time of the epoch
        endEpochTs = time.time() * 1000.0
        print(f"Epoch processing time: {(endEpochTs-startEpochTs)} ms")
        #After each epoch print a sample of the model's output
        generateAndPrintSample(model,tokenizer,device,startContext)

    #after all epochs return the training losses and validation losses
    return trainingLosses, validationLosses, trackTokensSeen    

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()



filePath = "input-data/contribution-critique-political-economy.txt"
#filePath = "input-data/the-verdict.txt"
#filePath = "processed-data/processed-text.txt"

with open(filePath, "r", encoding="utf-8") as f:
    textData = f.read()

tokenizer = GPT2Tokenizer()

totalCharacters = len(textData)
tokens = tokenizer.encode(textData)
totalTokens = len(tokens)

print(f"Characters {totalCharacters}")
print(f"Tokens {totalTokens}")

config = GPT_CONFIG_SMALL

trainRatio = 0.90
splitIdx = int(trainRatio * totalCharacters)
trainingData = textData[:splitIdx]
validationData = textData[splitIdx:]



# trainingDataLoader = createDataLoaderV1(
#     tokenizer=tokenizer,
#     text=trainingData, 
#     batchSize=2,
#     maxLength=config[CONTEXT_LENGTH],
#     stride=config[CONTEXT_LENGTH],
#     dropLast=True,
#     numWorkers=0
# )

# validationDataLoader = createDataLoaderV1(
#     tokenizer=tokenizer,
#     text=validationData, 
#     batchSize=2,
#     maxLength=config[CONTEXT_LENGTH],
#     stride=config[CONTEXT_LENGTH],
#     dropLast=True,
#     numWorkers=0
# )


trainingDataLoader = create_dataloader_v1(
    txt=trainingData,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=config[CONTEXT_LENGTH],
    stride=config[CONTEXT_LENGTH],
    drop_last=True,
    num_workers=0
)
validationDataLoader = create_dataloader_v1(
    txt=validationData,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=config[CONTEXT_LENGTH],
    stride=config[CONTEXT_LENGTH],
    drop_last=True,
    num_workers=0
)

device = torch.device("mps")
#device = torch.device("cpu")


print("\nStart training")
numberOfEpochs = 5
trainingLosses, validationLosses, tokensSeen = trainModel(
    modelName="TestCapital",
    modelConfig=GPT_CONFIG_SMALL,
    loadModelFromCheckpoint=True,
    trainingDataLoader=trainingDataLoader,
    validationDataLoader=validationDataLoader,
    learningRate=0.0004,
    weightDecay=0.1,
    device=device,
    numberOfEpochs=numberOfEpochs,
    evaluationFrequency=5,
    checkpointStorageFrequency=20,
    evaluationIterations=5,
    startContext="how does capital compare to labour",
    tokenizer=tokenizer
)

epochs_tensor = torch.linspace(0, numberOfEpochs, len(trainingLosses))
plot_losses(epochs_tensor, tokensSeen, trainingLosses, validationLosses)

