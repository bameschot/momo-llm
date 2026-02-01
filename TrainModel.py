import time
import math
import argparse
import pickle
import gzip
import random
import gc

import torch
import torch.nn as nn

from Tokenizers import *
from Data import create_tokenized_dataloader_v1,readInputFilePaths, TOKENIZER_PROCESSED_DATA_DIRECTORY
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
        description="Trains a momo-llm model"
    )
parser.add_argument("--model", type=str,default="TestEconomy_small_r", help="The name of the model to train")
parser.add_argument('--newModel', action='store_false',help= "indicates if a new model should be trained, if not the model file should be present in ./models/<model-name>/<model-name>.pth, a new model is stored in a folder with the same name")
parser.add_argument('--copyModel', type=str,default=None,help= "indicates if the model should be copied, the original model file should be present in ./models/<model-name>/<model-name>.pth, a new model has the indicated name postfixes (-<name>)")

parser.add_argument("--config", type=str,default="GPT_CONFIG_SMALL_CTX512_8_8_512", help="Determines the model configuration that a new model is initialised with, this parameter is ignored for models loaded from a checkpoint")
parser.add_argument("--inputData", type=str,default=TOKENIZER_PROCESSED_DATA_DIRECTORY, help="The glob pattern for selecting input data")
parser.add_argument('--shuffleInputFiles', action='store_true',help= "indicates if the (era) input files are shuffled")
parser.add_argument("--batchSize", type=int, default=40, help="The batch size of the dataloader")
parser.add_argument("--stride", type=int, default=None, help="The stride of the dataloader")

parser.add_argument("--minimalLearningRate", type=float, default=0.004, help="The lowest learning rate that the learning coefficient decay moves towards")
parser.add_argument("--peakLearningRate", type=float, default=0.001, help="The learning rate that the warmup increases towards")
parser.add_argument("--warmupSteps", type=int, default=1000, help="The amount of steps that the training spends in the warmup phase")
parser.add_argument("--weightDecay", type=float, default=0.0004, help="The decay in weights used by the AdamW optimizer")

parser.add_argument("--numberOfEpochs", type=int, default=10, help="The number of epochs (complete dataset passes) to train for")
parser.add_argument("--evaluationStepFrequency", type=int, default=200, help="The interval in steps for calculating and printing training progress")
parser.add_argument("--checkpointStepStorageFrequency", type=int, default=1000, help="The interval in steps for storing the <model-name>.pth and <model-name>.model subresults")
parser.add_argument("--evaluationIterations", type=int, default=10, help="The number of evaluations that is used to calculate the average loss over")
parser.add_argument("--startContext", type=str, default="What is a cat ", help="The example context used to generate the epoch example output")
parser.add_argument("--showLearningGraph", action='store_true',help="Indicates if the training loss and validation loss graph should be shown at the end of the training run")
parser.add_argument("--trainRatio", type=float,default=0.9,help="The training / validation data ratio taken from the dataset")
parser.add_argument('--compileModel', action='store_true',help= "indicates if a the model should be compiled")
parser.add_argument('--useAutocast', action='store_true',help= "indicates if a the model should use the scaler when trained (only supported on cuda)")
parser.add_argument('--moveDatasetToDevice', action='store_true',help= "indicates if the dataloaders should move all their data to the device, helps memory pressure on mps type devices")
parser.add_argument('--device', type=str, default=None, help= "indicates the device the model has to run on, if not provided the system autodetects in the order cuda->mps->cpu")
parser.add_argument('--shuffleBatches', action='store_true',help= "indicates if the batches in the dataloader should be shuffled")


args = parser.parse_args()

print(args.newModel)
p_inputData = args.inputData
p_shuffleInputFiles = args.shuffleInputFiles
p_batchSize=args.batchSize
p_stride = args.stride
p_trainRatio = args.trainRatio
p_config = args.config
p_model=args.model
p_loadModelFromCheckpoint=args.newModel
p_copyModel=args.copyModel
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
p_shuffleBatches=args.shuffleBatches
p_useAutocast = args.useAutocast
p_moveDatasetToDevice = args.moveDatasetToDevice
p_device = args.device

########################################
#Methods
########################################

def calculationLossBatch(inputBatch,targetBatch,model,device):
    #move the batches to the desired device
    if(inputBatch.device != device):
        inputBatch = inputBatch.to(device)
    if(targetBatch.device != device):
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
        print(f"generated text sample; {decodedText} == {str(tokenIds)}")
    model.train()

def trainModel(
        modelName,
        modelConfig,
        trainingDataLoader, 
        validationDataLoader, 
        device, 
        deviceName,
        tokenizer,
        loadModelFromCheckpoint=True,
        minimalLearningRate = 0.0001,
        peakLearningRate=0.0004,
        weightDecay=0.1,
        warmupSteps=100,
        numberOfEpochs = 10, 
        evaluationStepFrequency = 10, 
        evaluationIterations = 5,
        checkpointStepStorageFrequency = 100,
        startContext = "What is a cat ",
        compileModel = False,
        useAutocast = False,
        era = 1,
        model = None,
        optimizer = None
        ):          
    
    if(model is None or optimizer is None):
        model, optimizer, spModelBytes = loadModelForTraining(modelName,modelConfig,loadModelFromCheckpoint,peakLearningRate,weightDecay,compileModel)
        if spModelBytes is None:
            tokenizer = initializeTokenizer(trainingConfig[TOKENIZER_TYPE],trainingConfig[TOKENIZER_NAME])
        else:
            tokenizer = initializeTokenizerFromModelBytes(trainingConfig[TOKENIZER_TYPE],modelName,spModelBytes)
    
    return trainModelMedium(
        modelName=modelName,
        model=model,
        trainingDataLoader=trainingDataLoader,
        validationDataLoader=validationDataLoader,
        optimizer=optimizer,
        device=device,
        deviceName=deviceName,
        numberOfEpochs=numberOfEpochs,
        evaluationStepFrequency=evaluationStepFrequency,
        checkpointStepStorageFrequency=checkpointStepStorageFrequency,
        evaluationIterations=evaluationIterations,
        minimalLearningRate=minimalLearningRate,
        peakLearningRate=peakLearningRate,
        warmupSteps=warmupSteps,
        startContext=startContext,
        tokenizer=tokenizer,
        isCompiled=compileModel,
        useAutocast=useAutocast,
        era=era
    )
    # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

def trainModelMedium(
        modelName,
        model, 
        trainingDataLoader, 
        validationDataLoader, 
        optimizer, 
        device,
        deviceName, 
        tokenizer,
        minimalLearningRate = 0.0001,
        peakLearningRate=0.0004,
        warmupSteps=100,
        numberOfEpochs = 10, 
        evaluationStepFrequency = 10, 
        checkpointStepStorageFrequency = 100,
        evaluationIterations = 5,
        gradientClippingMaxNorm = 1.0,
        startContext = "What is a cat ",
        isCompiled=False,
        useAutocast=False,
        era=0
        ):
    #set start data
    tokensSeen, globalStep = -1,0
    totalTrainingSteps = numberOfEpochs * len(trainingDataLoader)
    learningRateIncrement = (peakLearningRate - minimalLearningRate) / warmupSteps
    learningRate = minimalLearningRate

    #iterate over the requested amount of epochs (complete batch runs)
    # Initialize the scaler
    if "cuda" in deviceName and useAutocast:
        scaler = torch.GradScaler()
    else:
        scaler = None
    
    for epoch in range(numberOfEpochs):
        startEpochTs = time.time() * 1000.0

        #iterate over the batches in steps
        epochSteps=0
        for inputBatch,targetBatch in trainingDataLoader:

            if epochSteps % 1000 == 0:
                # garbage collect & empty cuda/mps caches before each step
                if "cuda" in deviceName:
                    # torch.cuda.empty_cache()
                    None
                elif "mps" in deviceName:
                    # torch.mps.empty_cache()
                    None 
                gc.collect()

            epochSteps+=1
            startTs = time.time() * 1000.0
            #reset the loss gradients
            optimizer.zero_grad()

            if scaler != None:
                # use autocast to run the omptimizer at half precision
                with torch.autocast(device_type = deviceName, dtype = torch.bfloat16): 
                    #calculate the loss gradients for the batch
                    loss = calculationLossBatch(inputBatch,targetBatch,model,device)
                #execute the backwards pass (should be out of context)
                scaler.scale(loss).backward() # loss.backward()
            else: 
                #calculate the loss gradients for the batch
                loss = calculationLossBatch(inputBatch,targetBatch,model,device)
                loss.backward()

            #apply gradient clipping
            if(globalStep>warmupSteps):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradientClippingMaxNorm
                )

            #update the model weights with the loss gradients, use the scaler is availabe
            if scaler != None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            tokensSeen += inputBatch.numel()
            globalStep +=1

            #calculate the learning rate and top out after warmup and apply it to the optimizer
            if globalStep < warmupSteps:
                learningRate = minimalLearningRate+globalStep*learningRateIncrement
            else:
                #learningRate = peakLearningRate
                #apply cosine decay to the learning rate
                progress = ((globalStep - warmupSteps) / (totalTrainingSteps - warmupSteps))
                learningRate = minimalLearningRate + (peakLearningRate - minimalLearningRate) * 0.5 * (1 + math.cos(math.pi * progress))

            for parameterGroup in optimizer.param_groups:
                parameterGroup["lr"] = learningRate
            
            endTs = time.time() * 1000.0
            timePerStep=(endTs-startTs)/p_batchSize

            #if the current step reaches the evaluation fequency
            if(globalStep%evaluationStepFrequency==0):
                #evaluate the model and get the training loss and validation loss
                trainingLoss, validationLoss = evaluateModel(model,trainingDataLoader,validationDataLoader, device, evaluationIterations)

                #print the current results
                print(f"Epoch: [{era:02d}|{epoch:02d}/{numberOfEpochs:02d}] {(epochSteps/len(trainingDataLoader)*100):06.2f}%; (Step {globalStep:010d}): Training Loss {trainingLoss:.3f} Validation Loss {validationLoss:.3f}; avg. Step processing time {timePerStep:.3f} ms; learning rate: {learningRate:.10f}")
                

            #storage checkpoint interval
            if checkpointStepStorageFrequency is not None and globalStep%checkpointStepStorageFrequency==0:
                storeModel(modelName,model,isCompiled,tokenizer)
                storeCheckPoint(modelName,model,optimizer,isCompiled,tokenizer)
                print(f"Storing model: {modelName}")
                gc.collect()
    
        #end time of the epoch
        endEpochTs = time.time() * 1000.0
        print(f"Epoch [{epoch}] processing time: {(endEpochTs-startEpochTs)/3_600_000:.2f} hours")
        #After each epoch print a sample of the model's output
        generateAndPrintSample(model,tokenizer,device,startContext)
        
    #after all epochs return the training losses and validation losses
    storeModel(modelName,model,isCompiled,tokenizer)
    storeCheckPoint(modelName,model,optimizer,isCompiled,tokenizer)
    print(f"Storing model: {modelName}")
    return learningRate    

def loadModelForTraining(modelName, modelConfig,loadModelFromCheckpoint, learningRate, weightDecay, compileModel):
    if loadModelFromCheckpoint:
        model,optimizer,spModelBytes = loadCheckpoint(modelName,device,learningRate=learningRate,weightDecay=weightDecay)
        print(f"Loaded model {modelName} from file parameters: {model.numberOfParameters():_} and memory size: {model.memSizeMb():_} mb")
    else: 
        torch.set_default_dtype(getDataTypeFromConfig(modelConfig))
        model = GPTModel(modelConfig,device).to(device)
        print(f"Starting new model {modelName} with parameters: {model.numberOfParameters():_} and memory size: {model.memSizeMb():_} mb and config {model.config}")
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
        spModelBytes = None
    model.train()

    #compile the model if requested
    if compileModel:
        print("Compiling model")
        #required for mps compat
        #if  "is_available" in dir(torch.mps) and torch.mps.is_available():
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)#,mode="max-autotune")

    return model, optimizer, spModelBytes

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


def stepLearningRateForBatches(numFiles,peakLearningRate,minimalLearningRate):
    
    minimalLearningRates = []
    peakLearningRates = []
    learningRateStep = (peakLearningRate-minimalLearningRate) / numFiles

    for idx in range(numFiles):
        peakLearningRates.append(peakLearningRate-(learningRateStep*idx))
        minimalLearningRates.append(peakLearningRates[idx]-learningRateStep)
    
    return minimalLearningRates, peakLearningRates

########################################
#Script
########################################

if p_device is None:
    if torch.cuda.is_available():
       device = torch.device("cuda:0")
       deviceName = "cuda"
    elif torch.mps.is_available():
        device = torch.device("mps:0")
        deviceName = "mps"
    else:
        device = torch.device("cpu")
        deviceName = "cpu"
else:
    device = torch.device(p_device)
    deviceName = p_device

# garbage collect & empty cuda/mps caches before starting
if "cuda" in deviceName:
    torch.cuda.empty_cache()
elif "mps" in deviceName:
    torch.mps.empty_cache()
gc.collect()

trainingConfig = modelConfigs[p_config]


numDataLoaderWorkers = 0
inputPaths = readInputFilePaths(p_inputData)

if(p_shuffleInputFiles):
    random.shuffle(inputPaths)
else:
    inputPaths.sort()

print (f"Selected training files: {inputPaths}")

dataFileProcessedIdx = 0
model, optimizer, spModelBytes = loadModelForTraining(p_model,trainingConfig,p_loadModelFromCheckpoint,p_peakLearningRate,p_weightDecay,p_compileModel)

if spModelBytes is None:
    tokenizer = initializeTokenizer(trainingConfig[TOKENIZER_TYPE],trainingConfig[TOKENIZER_NAME])
else:
     tokenizer = initializeTokenizerFromModelBytes(trainingConfig[TOKENIZER_TYPE],p_model,spModelBytes)

if p_copyModel != None:
    p_model = p_model+f'-{p_copyModel}'

era = 0
for inputPath in inputPaths:

    print (f"Loading input: {inputPath}")

    #load and uncompress the datafiles, then extract the tokens
    with gzip.open(inputPath, "rb") as f:
        data = pickle.load(f)
    print(f"Input data loaded: {inputPath}")

    totalTokens = len(data)
    print (f"dataset total tokens: {totalTokens}")
    splitIdx = int(p_trainRatio * totalTokens)
    trainingData = data[:splitIdx]
    validationData = data[splitIdx:]

    #Print a sample
    print(f"training data sample;\n{tokenizer.decode(data[0:200])}") #\n{str(data[0:200])}")    

    #explicitly unload the data after it is put in the dataloader, slicing and loading copies the data
    del data
    gc.collect()

    #create the dataloaders for the input file
    dlStride = p_stride if p_stride != None else trainingConfig[CONTEXT_LENGTH]

    if "cuda" in deviceName or "cpu" in deviceName:
        dataloaderDtType = torch.long
    else:
        dataloaderDtType = torch.int

    trainingDataDevice = device if p_moveDatasetToDevice else None
    print(f"Loading data to device: {p_moveDatasetToDevice} -> {trainingDataDevice}")
    trainingDataLoader = create_tokenized_dataloader_v1(
        tokens=trainingData,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=dlStride,
        device=trainingDataDevice,
        dtType=dataloaderDtType,
        shuffle=p_shuffleBatches,
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )
    #explicitly unload the data after it is put in the dataloader, slicing and loading copies the data
    del trainingData
    gc.collect()
    print(f"Loaded training data")
    validationDataLoader = create_tokenized_dataloader_v1(
        tokens=validationData,
        batch_size=p_batchSize,
        max_length=trainingConfig[CONTEXT_LENGTH],
        stride=dlStride,
        device=trainingDataDevice,
        dtType=dataloaderDtType,
        shuffle=p_shuffleBatches,
        drop_last=True,
        num_workers=numDataLoaderWorkers
    )
    print(f"Loaded validation data")

    #explicitly unload the data after it is put in the dataloader, slicing and loading copies the data
    del validationData
    gc.collect()

    print(f"Dataloader config: batch size {p_batchSize} with stride {dlStride}")

    #start the training loop in this datafile , for the first loop check the parameter if a new model is needed and set the intial peak learning rate
    loadModelFromCheckpoint = True
    if dataFileProcessedIdx == 0:
        loadModelFromCheckpoint = p_loadModelFromCheckpoint

    print(f"\n[${era:02d}/{len(inputPaths):02d}]Start training for {inputPath} with peak learning rate: {p_peakLearningRate}")
    learningRate = trainModel(
        modelName=p_model,
        modelConfig=trainingConfig,
        loadModelFromCheckpoint=loadModelFromCheckpoint,
        trainingDataLoader=trainingDataLoader,
        validationDataLoader=validationDataLoader,
        tokenizer=tokenizer,
        minimalLearningRate =p_minimalLearningRate,
        peakLearningRate=p_peakLearningRate,
        warmupSteps=p_warmupSteps,    
        weightDecay=p_weightDecay,
        device=device,
        deviceName=deviceName,
        numberOfEpochs=p_numberOfEpochs,
        evaluationStepFrequency=p_evaluationStepFrequency,
        checkpointStepStorageFrequency=p_checkpointStepStorageFrequency,
        evaluationIterations=p_evaluationIterations,
        startContext=p_startContext,
        compileModel=p_compileModel,
        useAutocast=p_useAutocast,
        era=era,
        model=model,
        optimizer=optimizer
    )
    dataFileProcessedIdx += 1

    # explicitly unload the dataloaders
    del trainingDataLoader
    del validationDataLoader
    gc.collect()
    

    #tokensSeen.extend(runTokensSeen)
    #trainingLosses.extend(runTrainingLosses)
    #validationLosses.extend(runValidationLosses)
    era+=1


#if p_showLearningGraph:
#    epochs_tensor = torch.linspace(0, p_numberOfEpochs, len(trainingLosses))
#    plot_losses(epochs_tensor, tokensSeen, trainingLosses, validationLosses)


