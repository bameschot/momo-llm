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

def getDataTypeFromConfig(config): 
    # https://medium.com/data-science/pytorch-native-fp8-fedc06f1c9f7
    dt = torch.float32
    cfgDt = config[DEFAULT_DATA_TYPE]
    if cfgDt == 'bfloat16':
        dt = torch.bfloat16
    elif cfgDt == 'float16':
        dt = torch.float16
    elif cfgDt == 'float8_e4m3fn':
        dt = torch.float8_e4m3fn
    elif cfgDt == 'float8_e4m3fnuz':
        dt = torch.float8_e4m3fnuz
    elif cfgDt == 'float8_e5m2':
        dt = torch.float8_e5m2
    elif cfgDt == 'float8_e5m2fnuz':
        dt = torch.float8_e5m2fnuz
    elif cfgDt == 'float8_e8m0fnu':
        dt = torch.float8_e8m0fnu   

    return dt

def loadCheckpoint(modelName,device,learningRate=0.004,weightDecay=0.1):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}/{modelName}.pth",device)
    config = modelData["ModelConfig"]
    
    #this sets the default data type for all future operations based on config
    torch.set_default_dtype(getDataTypeFromConfig(config))

    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.train()

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
    optimizer.load_state_dict(modelData["OptimizerStateDict"])

    # training after loading is slower presumably due to this: https://github.com/Lightning-AI/pytorch-lightning/issues/19955
    for _, vv in optimizer.state.items():
        if "step" in vv and vv["step"].device.type != "cpu":
            vv["step"] = vv["step"].cpu()

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
    torch.set_default_dtype(getDataTypeFromConfig(config))

    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.eval()
    return model


#turns the keys of state dicts for compiled models to the keys for non-compiled models 
#based on https://github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089
def repairCompiledStateDict(stateDict):
    #print('Converting compiled state dictionary to uncompiled state dictionary')
    repairedDict = OrderedDict()
    for k,v in stateDict.items():
        newEntry = k.replace("_orig_mod.","")
        #print(f"Entry: {k} -> {newEntry}")
        repairedDict[newEntry] = v
    return repairedDict
