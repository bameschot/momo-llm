# How to run the training pipeline

## 1. (Optional) Add a new gpt config to the config file
- Go to `GPTModelConfig.py` and add a new model config definition and register it to the `modelConfigs` list
```
SIMPLE_ENG_4K_CONFIG_XXS_786_16L_14H = {
    CONFIG_NAME: "SIMPLE_ENG_4K_CONFIG_XXS_786_16L_14H",
    VOCABULARY_SIZE: 4000,
    CONTEXT_LENGTH: 786,
    EMBEDDING_DIMENSION: 70,
    N_HEADS: 14,
    N_LAYERS: 16,
    DROPOUT_EMBEDDING_RATE: 0.1,
    DROPOUT_ATTENTION_RATE: 0.1,
    DROPOUT_SHORTCUT_RATE: 0.1,
    QKV_BIAS: False,
    DEFAULT_DATA_TYPE: torch.bfloat16,
    TOKENIZER_TYPE: "sentencepiece",
    TOKENIZER_NAME: "wiki-simple-english-4k-lc-4000"
}
```
`"SIMPLE_ENG_4K_CONFIG_XXS_786_16L_14H": SIMPLE_ENG_4K_CONFIG_XXS_786_16L_14H,`


## 2. Train a vocabulary 
This step creates a new vocabulary file based on the provided input data of the requested size
- python3 PreProcessTokenizer.py --inputData="./input-data/wikiSimpleEnglish/**" --vocabSize=2000 --vocabularyName="wiki-simple-english-2k-lc" --forceLowerCase --newTrainingFile

## 3. Preprocess the dataset
This step pre-tokenized the requested data files using the indicated vocabulary and tokenizer
- python3 PreProcessDataFiles.py --inputData="./input-data/wikiSimpleEnglish/**|./input-data/tinyStories/**" --outputFileName="simple-english-2k-lc" --isTokenized --forceLowerCase --newOutputFile --outputBatchSizeMb=800 --vocabulary=simple-english-2k-lc-2000 --tokenizer=sentencepiece --newOutputFile 

## 3. Train the model
This step trains the model on the tokenized dataset 
- python3 TrainModel.py --modelName="simple-english-lc-2k-3m" --inputData="processed-data/simple-english-2k-lc/**.bin" --device="mps" --startContext="new york is the greatest city in america " --warmupSteps=2000 --evaluationStepFrequency=400 --checkpointStepStorageFrequency=1000 --batchSize=60  --numberOfEpochs=4 --peakLearningRate=0.005 --minimalLearningRate=0.0005 --trainingModelConfigName="SIMP_ENG_LC_2K_CONFIG_XS_786_10L_10H_150E" --newModel

For additional training runs remove the --newModel options. For non mps devices the model can be compiled and send to the cuda device with --compileModel and --sendToDevice