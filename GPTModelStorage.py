import torch

from GPTModelConfig import *
from GPTModel import GPTModel

MODEL_FOLDER = "./models"

def storeCheckPoint(modelName,model,optimizer):
    torch.save(
        {
            "ModelStateDict": model.state_dict(),
            "OptimizerStateDict": optimizer.state_dict()
        },
        f"{MODEL_FOLDER}/{modelName}/{modelName}.pth"
    )

def loadCheckpoint(modelName,config,device,learningRate=0.004,weightDecay=0.1):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}/{modelName}.pth",device)
    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=learningRate,weight_decay=weightDecay)
    optimizer.load_state_dict(modelData["OptimizerStateDict"])
    return model, optimizer

def storeModel(modelName,model):
    torch.save(
        {
            "ModelStateDict": model.state_dict()
        },
        f"{MODEL_FOLDER}/{modelName}/{modelName}.model"
    )

def loadModel(modelName,config,device):
    modelData = torch.load(f"{MODEL_FOLDER}/{modelName}/{modelName}.model",device)
    model = GPTModel(config).to(device)
    model.load_state_dict(modelData["ModelStateDict"])
    model.eval()
    return model
