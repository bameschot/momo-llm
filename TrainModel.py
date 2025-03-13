import time
import math
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim

from Data import GPT2Tokenizer, create_text_dataloader_v1,create_tokenized_dataloader_v1
from GenerateText import generateText, textToTokens, tokensToText
from GPTModel import GPTModel
from GPTModelConfig import *
from GPTModelStorage import *

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
        description="Trains a GPT model"
    )
parser.add_argument("--modelName", type=str,default="TestEconomy_small_r", help="The name of the model to train")
parser.add_argument('--newModel', action='store_false',help= "indicates if a new model should be trained, if not the model file should be present in ./models/<model-name>/<model-name>.pth, a new model is stored in a folder with the same name")
parser.add_argument("--trainingModelConfigName", type=str,default="GPT_CONFIG_SMALL_CTX512_8_8_512", help="Determines the model configuration that a new model is initialised with, this parameter is ignored for models loaded from a checkpoint")
parser.add_argument("--inputFilePath", type=str, default="processed-data/processed-text.txt", help="The path from where the raw text input data is loaded")
parser.add_argument("--inputFileIsTokenized", action='store_true',help="Indicates if the input file is tokenized with the gpt2 tokenizer")
parser.add_argument("--batchSize", type=int, default=8, help="The batch size of the dataloader")

parser.add_argument("--initialLearningRate", type=float, default=0.00001, help="The initial learning rate that is used when warming up the model")
parser.add_argument("--minimalLearningRate", type=float, default=0.0001, help="The lowest learning rate that the learning coefficient decay moves towards")
parser.add_argument("--peakLearningRate", type=float, default=0.0004, help="The learning rate that the warmup increases towards")
parser.add_argument("--warmupSteps", type=int, default=200, help="The amount of steps that the training spends in the warmup phase")
parser.add_argument("--weightDecay", type=float, default=0.1, help="The decay in weights used by the AdamW optimizer")

parser.add_argument("--numberOfEpochs", type=int, default=10, help="The number of epochs (complete dataset passes) to train for")
parser.add_argument("--evaluationStepFrequency", type=int, default=10, help="The interval in steps for calculating and printing training progress")
parser.add_argument("--checkpointStepStorageFrequency", type=int, default=100, help="The interval in steps for storing the <model-name>.pth and <model-name>.model subresults")
parser.add_argument("--evaluationIterations", type=int, default=5, help="The number of evaluations that is used to calculate the average loss over")
parser.add_argument("--startContext", type=str, default="What is a cat", help="The example context used to generate the epoch example output")
parser.add_argument("--showLearningGraph", action='store_true',help="Indicates if the training loss and validation loss graph should be shown at the end of the training run")
parser.add_argument("--trainRatio", type=float,default=0.9,help="The training / validation data ratio taken from the dataset")
parser.add_argument('--compileModel', action='store_true',help= "indicates if a the model should be compiled")
parser.add_argument('--device', type=str, default=None, help= "indicates the device the model has to run on, if not provided the system autodetects in the order cuda->mps->cpu")


args = parser.parse_args()

print(args.newModel)
p_inputFilePath = args.inputFilePath #"input-data/contribution-critique-political-economy.txt"
p_inputFileIsTokenized = args.inputFileIsTokenized
p_batchSize=args.batchSize
p_trainRatio = args.trainRatio
p_trainingConfigName = args.trainingModelConfigName
p_modelName=args.modelName
p_loadModelFromCheckpoint=args.newModel
p_initialLearningRate=args.initialLearningRate
p_minimalLearningRate =args.minimalLearningRate
p_peakLearningRate=args.peakLearningRate
p_warmupSteps=args.warmupSteps    
p_weightDecay=args.weightDecay
p_numberOfEpochs=args.numberOfEpochs
p_evaluationStepFrequency=args.evaluationStepFrequency
p_checkpointStepStorageFrequency=args.checkpointStepStorageFrequency
p_evaluationIterations=args.evaluationIterations
p_startContext=args.startContext
p_showLearningGraph=args.showLearningGraph
p_compileModel=args.compileModel
p_device = args.device

########################################
#Methods
########################################

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
        tokenIds = generateText(model,encoded,100)
        decodedText = tokensToText(tokenIds,tokenizer).replace("\n"," ")
        print(f"training text sample: {decodedText}")
    model.train()

def trainModel(
        modelName,
        modelConfig,
        trainingDataLoader, 
        validationDataLoader, 
        device, 
        tokenizer,
        loadModelFromCheckpoint=True,
        initialLearningRate=0.00001,
        minimalLearningRate = 0.0001,
        peakLearningRate=0.0004,
        weightDecay=0.1,
        warmupSteps=100,
        numberOfEpochs = 10, 
        evaluationStepFrequency = 10, 
        evaluationIterations = 5,
        checkpointStepStorageFrequency = 100,
        startContext = "What is a cat",
        compileModel = False,
        ):          
    if loadModelFromCheckpoint:
        model,optimizer = loadCheckpoint(modelName,device,learningRate=initialLearningRate,weightDecay=weightDecay)
        print(f"Loaded model {modelName} from file parameters: {model.numberOfParameters():_}")
    else: 
        model = GPTModel(modelConfig).to(device)
        print(f"Starting new model {modelName} with parameters: {model.numberOfParameters():_} and config {model.config}")
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=initialLearningRate,weight_decay=weightDecay)
    
    model.train()

    #compile the model if requested
    if compileModel:
        print("Compiling model")
        #required for mps compat
        if torch.mps.is_available():
            torch._dynamo.config.suppress_errors = True

        model = torch.compile(model,mode="max-autotune")
    
    print(f"Training model on {device}")

    
    return trainModelMedium(
        modelName=modelName,
        model=model,
        trainingDataLoader=trainingDataLoader,
        validationDataLoader=validationDataLoader,
        optimizer=optimizer,
        device=device,
        numberOfEpochs=numberOfEpochs,
        evaluationStepFrequency=evaluationStepFrequency,
        checkpointStepStorageFrequency=checkpointStepStorageFrequency,
        evaluationIterations=evaluationIterations,
        initialLearningRate=initialLearningRate,
        minimalLearningRate=minimalLearningRate,
        peakLearningRate=peakLearningRate,
        warmupSteps=warmupSteps,
        startContext=startContext,
        tokenizer=tokenizer
    )

def trainModelMedium(
        modelName,
        model, 
        trainingDataLoader, 
        validationDataLoader, 
        optimizer, 
        device, 
        tokenizer,
        initialLearningRate=0.00001,
        minimalLearningRate = 0.0001,
        peakLearningRate=0.0004,
        warmupSteps=100,
        numberOfEpochs = 10, 
        evaluationStepFrequency = 10, 
        checkpointStepStorageFrequency = 100,
        evaluationIterations = 5,
        startContext = "What is a cat"
        ):
    #set start data
    trainingLosses, validationLosses, trackTokensSeen = [],[],[]
    tokensSeen, globalStep = -1,0
    totalTrainingSteps = numberOfEpochs * len(trainingDataLoader)
    learningRateIncrement = (peakLearningRate - initialLearningRate) / warmupSteps
    learningRate = initialLearningRate

    #iterate over the requested amount of epochs (complete batch runs)
    for epoch in range(numberOfEpochs):
        model.train(),
        startEpochTs = time.time() * 1000.0

        #iterate over the batches in steps
        epochSteps=0
        for inputBatch,targetBatch in trainingDataLoader:
            epochSteps+=1
            startTs = time.time() * 1000.0
            #reset the loss gradients
            optimizer.zero_grad()
            #calculate the loss gradients for the batch
            loss = calculationLossBatch(inputBatch,targetBatch,model,device)
            #execute the backwards pass
            loss.backward()

            #apply gradient clipping
            if(globalStep>warmupSteps):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

            #update the model weights with the loss gradients
            optimizer.step()
            tokensSeen += inputBatch.numel()
            globalStep +=1

            #calculate the learning rate and top out after warmup and apply it to the optimizer
            if globalStep < warmupSteps:
                learningRate = initialLearningRate+globalStep*learningRateIncrement
            else:
                #learningRate = peakLearningRate
                #apply cosine decay to the learning rate
                progress = ((globalStep - warmupSteps) / (totalTrainingSteps - warmupSteps))
                learningRate = minimalLearningRate + (peakLearningRate - minimalLearningRate) * 0.5 * (1 + math.cos(math.pi * progress))

            for parameterGroup in optimizer.param_groups:
                parameterGroup["lr"] = learningRate
            
            endTs = time.time() * 1000.0
            timePerStep=(endTs-startTs)

            #if the current step reaches the evaluation fequency
            if(globalStep%evaluationStepFrequency==0):
                #evaluate the model and get the training loss and validation loss
                trainingLoss, validationLoss = evaluateModel(model,trainingDataLoader,validationDataLoader, device, evaluationIterations)
                trainingLosses.append(trainingLoss)
                validationLosses.append(validationLoss)

                #print the current results
                trackTokensSeen.append(tokensSeen)
                print(f"Epoch: {epoch:02d} {(epochSteps/len(trainingDataLoader)*100):06.2f}%; (Step {globalStep:010d}): Training Loss {trainingLoss:.3f} Validation Loss {validationLoss:.3f}; avg. Step processing time {timePerStep:.3f} ms; learning rate: {learningRate:.10f}")

            #storage checkpoint interval
            if checkpointStepStorageFrequency is not None and globalStep%checkpointStepStorageFrequency==0:
                storeModel(modelName,model)
                storeCheckPoint(modelName,model,optimizer)
                print(f"Storing model: {modelName}")
    
        #end time of the epoch
        endEpochTs = time.time() * 1000.0
        print(f"Epoch [{epoch}] processing time: {(endEpochTs-startEpochTs)} ms")
        #After each epoch print a sample of the model's output
        generateAndPrintSample(model,tokenizer,device,startContext)
        ##after each epoch plot progress
        #epochs_tensor = torch.linspace(0, numberOfEpochs, len(trainingLosses))
        #plot_losses(epochs_tensor, tokensSeen, trainingLosses, validationLosses)

    #after all epochs return the training losses and validation losses
    storeModel(modelName,model)
    storeCheckPoint(modelName,model,optimizer)
    print(f"Storing model: {modelName}")
    return trainingLosses, validationLosses, trackTokensSeen    

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

        #iterate over the batches in steps
        epochSteps=0
        for inputBatch,targetBatch in trainingDataLoader:
            epochSteps+=1
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
                print(f"Epoch: {epoch:02d} {(epochSteps/len(trainingDataLoader)*100):06.2f}%; (Step {globalStep:010d}): Training Loss {trainingLoss:.3f} Validation Loss {validationLoss:.3f} avg. Step processing time {timePerStep:.3f} ms")

            #storage checkpoint interval
            if checkpointStorageFrequency is not None and globalStep%checkpointStorageFrequency==0:
                storeModel(modelName,model)
                storeCheckPoint(modelName,model,optimizer)
                print(f"Storing model: {modelName}")
    
        #end time of the epoch
        endEpochTs = time.time() * 1000.0
        print(f"Epoch [{epoch}] processing time: {(endEpochTs-startEpochTs)} ms")
        #After each epoch print a sample of the model's output
        generateAndPrintSample(model,tokenizer,device,startContext)
        ##after each epoch plot progress
        if p_showLearningGraph:
            try:
                epochs_tensor = torch.linspace(0, numberOfEpochs, len(trainingLosses))
                plot_losses(epochs_tensor, tokensSeen, trainingLosses, validationLosses)
            except:
                pass

    #after all epochs return the training losses and validation losses
    storeModel(modelName,model)
    storeCheckPoint(modelName,model,optimizer)
    print(f"Storing model: {modelName}")
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

########################################
#Script
########################################

if p_device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(p_device)

trainingConfig = modelConfigs[p_trainingConfigName]


tokenizer = GPT2Tokenizer()


numDataLoaderWorkers = 0

if p_inputFileIsTokenized:
    with open(p_inputFilePath, "rb") as f:
        data = pickle.load(f)
    print(f"Input data loaded: {p_inputFilePath}")
else:
    with open(p_inputFilePath, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"Input data loaded: {p_inputFilePath}")


totalCharacters = len(data)
splitIdx = int(p_trainRatio * totalCharacters)
trainingData = data[:splitIdx]
validationData = data[splitIdx:]


if p_inputFileIsTokenized:
    trainingDataLoader = create_tokenized_dataloader_v1(
        tokens=trainingData,
        tokenizer=tokenizer,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=trainingConfig[CONTEXT_LENGTH],
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )
    validationDataLoader = create_tokenized_dataloader_v1(
        tokens=validationData,
        tokenizer=tokenizer,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=trainingConfig[CONTEXT_LENGTH],
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )
else:    
    trainingDataLoader = create_text_dataloader_v1(
        txt=trainingData,
        tokenizer=tokenizer,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=trainingConfig[CONTEXT_LENGTH],
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )
    validationDataLoader = create_text_dataloader_v1(
        txt=validationData,
        tokenizer=tokenizer,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=trainingConfig[CONTEXT_LENGTH],
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )

print("\nStart training")
print(p_loadModelFromCheckpoint)
trainingLosses, validationLosses, tokensSeen = trainModel(
    modelName=p_modelName,
    modelConfig=trainingConfig,
    loadModelFromCheckpoint=p_loadModelFromCheckpoint,
    trainingDataLoader=trainingDataLoader,
    validationDataLoader=validationDataLoader,
    tokenizer=tokenizer,
    initialLearningRate=p_initialLearningRate,
    minimalLearningRate =p_minimalLearningRate,
    peakLearningRate=p_peakLearningRate,
    warmupSteps=p_warmupSteps,    
    weightDecay=p_weightDecay,
    device=device,
    numberOfEpochs=p_numberOfEpochs,
    evaluationStepFrequency=p_evaluationStepFrequency,
    checkpointStepStorageFrequency=p_checkpointStepStorageFrequency,
    evaluationIterations=p_evaluationIterations,
    startContext=p_startContext,
    compileModel=p_compileModel
)

if p_showLearningGraph:
    epochs_tensor = torch.linspace(0, p_numberOfEpochs, len(trainingLosses))
    plot_losses(epochs_tensor, tokensSeen, trainingLosses, validationLosses)

