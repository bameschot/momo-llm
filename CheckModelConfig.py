import argparse
from GPTModel import GPTModel
from GPTModelConfig import *

########################################
#Parameters
########################################
parser = argparse.ArgumentParser(
    description="Trains a momo-llm model"
)

parser.add_argument("--config", type=str,default=None, help="Determines the model configuration that a new model is initialised with, this parameter is ignored for models loaded from a checkpoint")

args = parser.parse_args()
p_config = args.config

########################################
#Scripts
########################################

if p_config is not None:
    config = modelConfigs[p_config]
    model = GPTModel(config,'cpu')
    print(f"Starting new model {p_config} with parameters: {model.numberOfParameters():_} and memory size: {model.memSizeMb():_} mb and config {model.config}")

     