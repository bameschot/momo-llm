import torch
import torch.nn as nn

from Preprocessing import GPT2Tokenizer, createDataLoaderV1, create_dataloader_v1
from GPTModel import GPTModel
from GPTModelConfig import *

torch.manual_seed(123)


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

def trainModelSimple(
        model, 
        trainingDataLoader, 
        validationDataLoader, 
        optimizer, 
        device, 
        numberOfEpochs, 
        evaluationFrequency, 
        evaluationIterations,
        startContext,
        tokenizer):
    #set start data
    trainingLosses, validationLosses, trackTokensSeen = [],[],[]
    tokensSeen, globalStep = -1,0

    #iterate over the requested amount of epochs (complete batch runs)
    for epoch in range(numberOfEpochs):
        model.train(),
        #iterate over the batches
        for inputBatch,targetBatch in trainingDataLoader:
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

            #if the current step reaches the evaluation fequency
            if(globalStep%evaluationFrequency==0):
                #evaluate the model and get the training loss and validation loss
                trainingLoss, validationLoss = evaluateModel(model,trainingDataLoader,validationDataLoader, device, evaluationIterations)
                trainingLosses.append(trainingLoss)
                validationLosses.append(validationLoss)

                #print the current results
                trackTokensSeen.append(tokensSeen)
                print(f"Epoch: {epoch}; (Step {globalStep:06d}): Training Loss {trainingLoss:.3f} Validation Loss {validationLoss:.3f}")
    
        #After each epoch print a sample of the model's output
        generateAndPrintSample(model,tokenizer,device,startContext)

    #after all epochs return the training losses and validation losses
    trainingLosses, validationLosses, trackTokensSeen    



filePath = "input-data/the-verdict.txt"
with open(filePath, "r", encoding="utf-8") as f:
    textData = f.read()

tokenizer = GPT2Tokenizer()

totalCharacters = len(textData)
tokens = tokenizer.encode(textData)
totalTokens = len(tokens)

print(f"Characters {totalCharacters}")
print(f"Tokens {totalTokens}")

config = GPT_CONFIG_SMALL_CTX256

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
    drop_last=False,
    num_workers=0
)
validationDataLoader = create_dataloader_v1(
    txt=validationData,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=config[CONTEXT_LENGTH],
    stride=config[CONTEXT_LENGTH],
    drop_last=False,
    num_workers=0
)

print("\ntraining loader")
for x,y in trainingDataLoader:
    print(x.shape,y.shape)

print("\nvalidation loader")
for x,y in validationDataLoader:
    print(x.shape,y.shape)

device = torch.device("mps")
device = torch.device("cpu")

model = GPTModel(config).to(device)

with torch.no_grad():
    trainingLoss = calculationLossLoader(trainingDataLoader,model,device)
    validationLoss = calculationLossLoader(validationDataLoader,model,device)

print(f"training loss: {trainingLoss}")
print(f"validation loss: {validationLoss}")

