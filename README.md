# sequence_analyzer

Analyze medical sequence from cdm_datasetmaker-datasets by deep learning methods
1. Building, training and testing sequential models
2. Supports gird-search method for hyper-parameter tuning
3. Reporting logs, fiugres and tables
4. Monitoring by tensorboard

## Dependency

Install cdm_datasetmaker:
```sh
git clone https://github.com/DannielJin/cdm_datasetmaker.git
pip3 install cdm_datasetmaker/dist/*.whl
```

## Installation

At /PATH/TO/THE/PACKAGE/FOLDER/:

```sh
pip3 install ./dist/*.whl
```
```sh
pip3 uninstall ./dist/*.whl -y
```

## Usage example

Make SA_PARAMS.txt in CONFIG FOLDER:
```
## (e.g.)
## multiple items for grid search
MODEL_ARCH = RNN_MODEL, RNN_ATTENTION_MODEL, BIRNN_ATTENTION_MODEL ##RNN_MODEL, RNN_ATTENTION_MODEL, BIRNN_ATTENTION_MODEL
BATCH_SIZE = 128
EMB_MATRIX_FILENAME = False, True   ## If True, 'True' will be replaced by the *_emb_matrix_*.pkl filenames.
EMB_SIZE = 64, 128
ATT_H_SIZE = 32  # attention_layer_hidden_size when MODEL_ARCH == *_ATTENTION_*

CLASS_WEIGHT = 6.0  ## panelty when loss occurs with target class
KEEP_PROB = 0.5
L2_REG = 1e-3   ## For ridge regularization terms
LR = 1e-2, 5e-4
DECAY_STEPS = 100
DECAY_RATE = 0.9
TRAIN_STEPS = 1000
PRINT_BY = 10   ## printing and logging to tensorboard
SAVE_BY = 1000 

## RNN_ARCH; you can build multiple models by setting 'RNN_ARCH_{}_'...
# rnn_architecture #1
RNN_ARCH_1_cell_type = GRU  # GRU, LSTM
RNN_ARCH_1_hidden_size = 64  # first rnn layer with 64 hidden_size
RNN_ARCH_1_drop_out = False  # first rnn layer with no drop_out unit
# rnn_architecture #2
RNN_ARCH_2_cell_type = GRU  # GRU, LSTM
RNN_ARCH_2_hidden_size = 64, 128   # first rnn layer with 64 hidden_size & second one with 128 hidden_size
RNN_ARCH_2_drop_out = False, True  # first rnn layer with no drop_out unit & second one with drop_out unit
```

Main codes:
1. get datasets by cdm_datasetmaker:
```
from cdm_datasetmaker import Get_datasets
datasets = Get_datasets(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',       #/PATH/TO/CONFIG/FOLDER/
                        DATA_FOLDER_PATH = '/PRJ/DATA/',           #/PATH/TO/DATA/FOLDER/ (save&load)
                        RESULT_FOLDER_PATH = '/PRJ/RESULT/',       #/PATH/TO/RESULT/FOLDER/ (logs)
                        PROJECT_NAME = 'PROJECT_DATASETS',         #PROJECT_NAMES
                        DB_CONN_FILENAME = 'DB_connection.txt',
                        DS_PARAMS_FILE_NAME = 'DS_PARAMS.txt', 
                        PIPELINE_START_LEVEL = 4)                  #Starting level
```
PIPELINE_START_LEVEL; 
    1. Make_target_comp_tables  (when the first time)
    2. Table to rawSeq
    3. RawSeq to multihot
    4. Multihot to Dataset      (when you want to restore datasets)

2. (OPTION) make emb_matrix by medterm2vec:
```
from medterm2vec import Run
df_emb_results = Run(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',          #/PATH/TO/CONFIG/FOLDER/
                     DATA_FOLDER_PATH = '/PRJ/DATA/',              #/PATH/TO/DATA/FOLDER/ (cdm_datasetmaker)
                     RESULT_FOLDER_PATH = '/PRJ/RESULT/',          #/PATH/TO/RESULT/FOLDER/ (logs, model save&load)
                     PROJECT_NAME = 'PROJECT_EMB',                 #PROJECT_NAMES
                     EMB_PARAMS_FILE_NAME = 'EMB_PARAMS.txt', 
                     DATASETS_INFO = datasets.info,                #datasets.info from the datasets object
                     SKIP_EDGE_EXTRACTING = True,                  #If False, edge_extracting process will be started
                     NEW_GAME = True)                              #Fresh starting for training embedding models 
```

3. Run pipeline:
```
%matplotlib inline
```
%matplotlib inline -> when you run the codes in jupyter notebook

```
from sequence_analyzer import Run
df_results = Run(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',              #/PATH/TO/CONFIG/FOLDER/
                 RESULT_FOLDER_PATH = '/PRJ/RESULT/',              #/PATH/TO/RESULT/FOLDER/ (logs, model save&load)
                 PROJECT_NAME = 'PROJECT_ANALYSIS',                #PROJECT_NAMES
                 SA_PARAMS_FILE_NAME = 'SA_PARAMS.txt', 
                 DATASETS = datasets,                              #datasets object
                 NEW_GAME = True,                                  #Fresh starting; saved models will be removed
                 TEST_ONLY = False,                                #IF True, training will be skipped
                 MODEL_INDICES = [-1])                             #The model_indices of model_list to run. [-1] for all.
```

(e.g.) If you want to keep training MODEL_1, MODEL_3 only,
```
df_results = Run(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',              
                 RESULT_FOLDER_PATH = '/PRJ/RESULT/',              
                 PROJECT_NAME = 'PROJECT_ANALYSIS',                
                 SA_PARAMS_FILE_NAME = 'SA_PARAMS.txt', 
                 DATASETS = datasets,                              
                 NEW_GAME = False,                                  
                 TEST_ONLY = False,                                
                 MODEL_INDICES = [1, 3]) 
```

When you run the pipeline, tensorboard service will be started. 
If you run the codes in jupyter notebook, click the tensorboard_service_address (printed)
Or, you can run tensorboard manually.
```
from sequence_analyzer.report import Run_tensorboard
Run_tensorboard('/RESULT_FOLDER_PATH/PROJECT_NAME/CDM_DB_NAME/')
```
```
from sequence_analyzer.report import Stop_tensorboard
Stop_tensorboard()
```

## Release History

* 1.0.0
    * released

## Meta

Sanghyung Jin, MS(1) – jsh90612@gmail.com  
Yourim Lee, BS(1) - urimeeee.e@gmail.com  
Rae Woong Park, MD, PhD(1)(2) - rwpark99@gmail.com  

(1) Dept. of Biomedical Informatics, Ajou University School of Medicine, Suwon, South Korea  
(2) Dept. of Biomedical Sciences, Ajou University Graduate School of Medicine, Suwon, South Korea  




