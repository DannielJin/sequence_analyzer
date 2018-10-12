def get_logger_instance(logger_name, DUMPING_PATH, parent_name=False, stream=False):
    import logging, os, sys, datetime
    if parent_name:
        logger = logging.getLogger(parent_name+'.'+logger_name)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    #stream_handler
    if stream:
        stream_hander = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_hander)
    
    #file_handler  
    if parent_name:
        logFilePath = os.path.join(DUMPING_PATH, parent_name+'_'+logger_name+'.log')
    else:
        logFilePath = os.path.join(DUMPING_PATH, logger_name+'.log')
    file_handler = logging.FileHandler(filename=logFilePath)
    logger.addHandler(file_handler)
    
    if parent_name==False:
        logger.info("\n\n" + "@"*100 + "\n" + "@"*100)
        logger.info("\n{}".format(datetime.datetime.now()))
        logger.info("\n[Start Logging..]\n\n")    
    return logger


def dumpingFiles(logger, filePath, outFilename, files, verbose=True):
    """
    If you don't want to log, set logger to None
    """
    import os, pickle
    dumpingPath = os.path.join(filePath, outFilename)
    if verbose:
        if logger is None:
            print("Dumping at..", dumpingPath)
        else:
            logger.info("Dumping at.. {}".format(dumpingPath))
    with open(dumpingPath, 'wb') as outp:
        pickle.dump(files, outp, -1)
        
def loadingFiles(logger, filePath, filename, verbose=True):
    """
    If you don't want to log, set logger to None
    """
    import os, pickle
    loadingPath = os.path.join(filePath, filename)
    #print("Loading at..", loadingPath)
    if verbose: 
        if logger is None:
            print("Loading at.. {}".format(loadingPath))
        else:
            logger.info("Loading at.. {}".format(loadingPath))
    with open(loadingPath, 'rb') as f:
        p = pickle.load(f)
    return p

def option_printer(logger, **kwargs):
    logger.info("{0:>26}   {1:}".format('[OPTION]', '[VALUE]'))
    for k in sorted(kwargs.keys()):
        if k!='EMB_MATRIX':
            logger.info("  {0:>23}:   {1:}".format(k, kwargs[k]))
            
def get_param_dict(FILE_NAME, CONFIG_FOLDER_PATH):
    import os, re
    FILE_PATH = os.path.join(CONFIG_FOLDER_PATH, FILE_NAME)
    param_dict = dict()
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try: #remove comments
                cut_idx = re.search('#.*', line).start()
                line = line[:cut_idx]
            except:
                pass
            
            particles = [p.strip() for p in line.split('=', maxsplit=1)]
            if len(particles)==1:
                continue
            key = particles[0]
            val = particles[1:] 
            
            if ',' in val[0]:
                val = [v.strip() for v in val[0].split(',')]
            try: val = [int(v) for v in val]
            except:
                try: val = [float(v) for v in val]
                except:
                    val = [True if v.lower()=='true' else False if v.lower()=='false' else v for v in val]
            param_dict[key] = val
    return param_dict
