
def Run(**kwargs): 
    """
    [Essential]
      "Load params from 'CONFIG_FOLDER_PATH/SA_PARAMS_FILE_NAME'"
    CONFIG_FOLDER_PATH; (e.g.) '/path/to/CONFIG'
    RESULT_FOLDER_PATH; (e.g.) '/path/to/RESULT'
    PROJECT_NAME; (e.g.) 'project'
    SA_PARAMS_FILE_NAME; (e.g.) 'SA_PARAMS.txt'
    DATASETS; a dataset object
    NEW_GAME; (e.g) False 
    TEST_ONLY; (e.g) False
    MODEL_INDICES; (e.g) default: [-1]; Use all the models.
    
    ######################### ALL PARAMS ##############################
    
    [basics]
    CONFIG_FOLDER_PATH; (e.g.) '/path/to/CONFIG'
    RESULT_FOLDER_PATH; (e.g.) '/path/to/RESULT'
    PROJECT_NAME; (e.g.) 'project'
    SA_PARAMS_FILE_NAME; (e.g.) 'SA_PARAMS.txt'
    
    [runtime_params]
    DATASETS; a dataset object
    NEW_GAME; (e.g) False 
    TEST_ONLY; (e.g) False
    
    [model_params]
    MODEL_ARCH; (e.g.) RNN_MODEL
    BATCH_SIZE; (e.g.) 128
    EMB_MATRIX_FILENAME; (e.g.) [False, True] # True will be replaced by all of emb_matrices in DATA_FOLDER_PATH
    EMB_SIZE; (e.g.) 64, 128
    
    KEEP_PROB; (e.g.) 0.5
    L2_REG; (e.g.) 0.001
    LR; (e.g.) 0.0005
    DECAY_RATE; (e.g.) 0.9
    DECAY_STEPS; (e.g.) 1000
    TRAIN_STEPS; (e.g.) 1000
    PRINT_BY; (e.g.) 2000
    SAVE_BY; (e.g.) 1000000
    
    ## RNN_ARCH; you can build multiple models by setting RNN_ARCH_{}_...
    RNN_ARCH_1_cell_type; (e.g.) 'GRU'
    RNN_ARCH_1_drop_out; (e.g.) [False, True]
    RNN_ARCH_1_hidden_size; (e.g.) [32, 64]
    
    RNN_ARCH_2_cell_type; (e.g.) 'GRU'
    RNN_ARCH_2_drop_out; (e.g.) [True, True]
    RNN_ARCH_2_hidden_size; (e.g.) [64, 128]
    
    """
    from .utils import get_logger_instance, option_printer, loadingFiles, get_param_dict
    from .model import Get_model_list
    from .train import Train_model_list, Test_model_list
    from .report import Get_datasets_info
    import os, glob
    import logging, datetime
    from importlib import reload
    
    ## get params
    try: param_dict = get_param_dict(kwargs['SA_PARAMS_FILE_NAME'], kwargs['CONFIG_FOLDER_PATH'])
    except: param_dict = get_param_dict(kwargs['SA_PARAMS_FILE_NAME'], kwargs['DATASETS'].info['CONFIG_FOLDER_PATH'])
    param_dict.update(kwargs)
    if 'MODEL_INDICES' not in param_dict.keys():
        param_dict['MODEL_INDICES'] = [-1] #Use all.
    
    param_dict['DUMPING_PATH'] = os.path.join(param_dict['RESULT_FOLDER_PATH'], 
                                              param_dict['PROJECT_NAME'], 
                                              param_dict['DATASETS'].info['CDM_DB_NAME'])
    
    if param_dict['NEW_GAME']:
        if param_dict['TEST_ONLY']:
            print("\n\t(NEW_GAME => False)")
            param_dict['NEW_GAME'] = False   
        else:
            print("[!!] Are you sure NEW_GAME is True?; \n\t(REMOVE ALL RESULTS AND START OVER)")
            confirm = input()
            if confirm.lower() in ['y', 'yes', 'true']:
                print("\n\t(NEW_GAME => True)")
                import shutil, glob, os
                _ = [shutil.rmtree(p) for p in glob.glob(param_dict['DUMPING_PATH'])] #remove param_dict['DUMPING_PATH']
            else:
                print("\n\t(NEW_GAME => False)")
                param_dict['NEW_GAME'] = False            
        
    if not os.path.exists(param_dict['DUMPING_PATH']): 
        os.makedirs(param_dict['DUMPING_PATH'])

    ## logger
    logging.shutdown()
    reload(logging)
    main_logger = get_logger_instance(logger_name='sa_pipeline', 
                                      DUMPING_PATH=param_dict['DUMPING_PATH'], 
                                      parent_name=False,
                                      stream=True)
    
    ## EMB_MATRIX_FILENAME (True -> EMB_MATRIX_FILENAMEs)
    if True in param_dict['EMB_MATRIX_FILENAME']:
        param_dict['EMB_MATRIX_FILENAME'] = [v for v in param_dict['EMB_MATRIX_FILENAME'] if v!=True] #remove True
        emb_filepath_list = [p.split('/')[-1] for p in glob.glob(param_dict['DATASETS'].info['DATA_FOLDER_PATH']+'*emb*')]
        param_dict['EMB_MATRIX_FILENAME'] += emb_filepath_list #add emb_filepath_list
    
    ## print options
    main_logger.info("\n (params) \n")
    option_printer(main_logger, **param_dict)
    main_logger.info("="*100 + "\n")
    
    ## [1] Inspect datasets
    main_logger.info("\n[Datasets Info.]\n")
    option_printer(main_logger, **param_dict['DATASETS'].info)
    
    Get_datasets_info(main_logger, param_dict['DATASETS'], 'TRAIN', 'TARGET', topK = 15, thr = 0.1)
    Get_datasets_info(main_logger, param_dict['DATASETS'], 'TRAIN', 'COMP', topK = 15, thr = 0.1)
    Get_datasets_info(main_logger, param_dict['DATASETS'], 'TEST', 'TARGET', topK = 15, thr = 0.1)
    Get_datasets_info(main_logger, param_dict['DATASETS'], 'TEST', 'COMP', topK = 15, thr = 0.1)
    main_logger.info("="*100 + "\n")
    
    ## [2] Make Models
    main_logger.info("\n[Model Setting]\n")
    model_list = Get_model_list(param_dict, param_dict['DUMPING_PATH'], model_indices=param_dict['MODEL_INDICES'])
    main_logger.info("="*100 + "\n")
    
    ## [3] Train Models
    if not param_dict['TEST_ONLY']:
        Train_model_list(MODEL_LIST = model_list,
                         DATASETS = param_dict['DATASETS'],
                         DUMPING_PATH = param_dict['DUMPING_PATH'], 
                         new_game = param_dict['NEW_GAME'])
        main_logger.info("="*100 + "\n")
    
    ## [4] Test Models
    Test_model_list(MODEL_LIST = model_list,
                    DATASETS = param_dict['DATASETS'],
                    DUMPING_PATH = param_dict['DUMPING_PATH'])
    main_logger.info("="*100 + "\n")
    
    ## [5] Get Results
    main_logger.info("\n[Model_results]\n")
    df_results = loadingFiles(main_logger, param_dict['DUMPING_PATH'], 'df_RESULTS.pkl')
    
    main_logger.info("\nALL DONE!!")
    for h in list(main_logger.handlers):
        main_logger.removeHandler(h)
        h.flush()
        h.close()
    return df_results
    