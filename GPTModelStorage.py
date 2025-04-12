from pathlib import Path
from collections import OrderedDict

import torch

from GPTModelConfig import *
from GPTModel import GPTModel

MODEL_FOLDER = "./models"

def storeCheckPoint(modelName,model,optimizer,isCompiled=False):
    Path(f"{MODEL_FOLDER}/{modelName}").mkdir(parents=True, exist_ok=True)
    
    modelStateDict = repairCompiledStateDict(model.state_dict()) if isCompiled else model.state_dict()
    optimizerStateDict = repairCompiledStateDict(optimizer.state_dict()) if isCompiled else optimizer.state_dict()

    torch.save(
        {
            "ModelConfig": model.config,
            "ModelStateDict": modelStateDict,
            "OptimizerStateDict": optimizerStateDict
        },
        f"{MODEL_FOLDER}/{modelName}/{modelName}.pth"
    )

def loadCheckpoint(modelName,device,learningRate=0.004,weightDecay=0.1):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}/{modelName}.pth",device)
    config = modelData["ModelConfig"]
    
    #this sets the default data type for all future operations based on config
    torch.set_default_dtype(config[DEFAULT_DATA_TYPE])

    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
    optimizer.load_state_dict(modelData["OptimizerStateDict"])
    return model, optimizer

def storeModel(modelName,model,isCompiled=False):
    Path(f"{MODEL_FOLDER}/{modelName}").mkdir(parents=True, exist_ok=True)

    modelStateDict = repairCompiledStateDict(model.state_dict()) if isCompiled else model.state_dict()

    torch.save(
        {
            "ModelConfig": model.config,
            "ModelStateDict": modelStateDict
        },
        f"{MODEL_FOLDER}/{modelName}/{modelName}.model"
    )

def loadModel(modelName,device):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}/{modelName}.model",device)
    config = modelData["ModelConfig"]

    #this sets the default data type for all future operations based on config
    torch.set_default_dtype(config[DEFAULT_DATA_TYPE])

    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.eval()
    return model


#turns the keys of state dicts for compiled models to the keys for non-compiled models 
#based on https://github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089
def repairCompiledStateDict(stateDict):
    print('Converting compiled state dictionary to uncompiled state dictionary')
    repairedDict = OrderedDict()
    for k,v in stateDict.items():
        newEntry = k.replace("_orig_mod.","")
        #print(f"Entry: {k} -> {newEntry}")
        repairedDict[newEntry] = v
    return repairedDict
