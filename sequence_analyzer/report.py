
from .utils import loadingFiles

def get_group_indices(pred, labels):
    import numpy as np
    g_treat = labels[:,1]==1
    g_comp = labels[:,0]==1
    g_pos = np.argmax(pred, axis=1)==1
    g_neg = np.argmax(pred, axis=1)==0

    g_true_positive = g_treat * g_pos
    g_true_negative = g_comp * g_neg
    g_false_positive = g_comp * g_pos
    g_false_negative = g_treat * g_neg
    return g_true_positive, g_true_negative, g_false_positive, g_false_negative

def get_attention_codes(inputs, alphas, idx2code, topK_time=1, topK_code=15):
    import numpy as np
    from collections import Counter
    collection = []
    for p_inputs, p_alpha in zip(inputs, alphas):
        time_indices = np.argsort(p_alpha)[::-1][:topK_time]
        code_indices = np.nonzero(p_inputs[time_indices])[1]
        collection.append(code_indices)
    att_codes = Counter([c for p in collection for c in p]).most_common(topK_code)
    return [(idx2code[idx], freq) for idx, freq in att_codes]

def draw_roc_curve(logPath, true, pred):
    """
    (ONLY PLOT for label 1)
    logPath: where to save the figure
    true: numpy array. shape; (B, Label_size)
    pred: numpy array. shape; (B, Label_size)
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, auc

    colors = ['darkorange', 'cornflowerblue', 'cornflowerblue', 'teal']
    
    plt.figure(figsize=(10,7))
    fpr, tpr, _ = roc_curve(y_true=true[:, 1], y_score=pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[0], lw=2, label="ROC curve (area = {0:0.3f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(logPath+'_AUROC.png', bbox_inches='tight')
    plt.show()
    
def draw_pr_curve(logPath, true, pred):
    """
    (ONLY PLOT for label 1)
    logPath: where to save the figure
    true: numpy array. shape; (B, Label_size)
    pred: numpy array. shape; (B, Label_size)
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    colors = ['darkorange', 'cornflowerblue', 'cornflowerblue', 'teal']

    plt.figure(figsize=(10,7))
    precision, recall, _ = precision_recall_curve(y_true=true[:, 1], probas_pred=pred[:, 1])
    prc_auc = average_precision_score(y_true=true[:, 1], y_score=pred[:, 1])
    plt.plot(recall, precision, color=colors[0], lw=2, label="PR curve (area = {0:0.3f})".format(prc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig(logPath+'_AUPRC.png', bbox_inches='tight')
    plt.show()

def jaccard_index(A, B):
    import numpy as np
    union = np.union1d(A, B).shape[0]
    if union==0: return 0.0
    else: return np.intersect1d(A, B).shape[0] / union

def statsInfo(logger, seq_data, demo_data, seq_len):
    logger.info("\n  <statsInfo>")
    if type(seq_data)!=list:
        seq_data = [seq_data]
    if type(demo_data)!=list:
        demo_data = [demo_data]
    if type(seq_len)!=list:
        seq_len = [seq_len]
        
    male_count = sum([p[0][0] for p in demo_data])
    avg_age_data = [sum([v[1] for v in p])/len(p) for p in demo_data]
    coCode_freq = [len(row.indices) for sprs_m, l in zip(seq_data, seq_len) for row in sprs_m[:l]]
    
    from scipy import stats
    stats_name = ['nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis']
    logger.info("\n  {0};".format('[gender level]'))
    for k, v in zip(['male', 'female', 'SUM'], [male_count, len(demo_data)-male_count, len(demo_data)]):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0};".format('[avg_age info]'))
    for k, v in zip(stats_name, stats.describe(avg_age_data)):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0} sequence length;".format('[visit level]'))
    for k, v in zip(stats_name, stats.describe(seq_len)):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0} # of co-code;".format('[code level]'))
    for k, v in zip(stats_name, stats.describe(coCode_freq)):
        logger.info("  {0:>12}: {1}".format(k, v))
        
def topK_codeFrequency(logger, DATA_PATH, seq_data, topK):
    logger.info("\n  <topK_codeFrequency>\n")
    code2title = loadingFiles(logger, DATA_PATH, 'code2title.pkl')
    code2idx = loadingFiles(logger, DATA_PATH, 'code2idx.pkl')
    idx2code = {v:k for k,v in code2idx.items()}
    
    if type(seq_data)!=list:
        seq_data = [seq_data]
    from collections import Counter
    c = Counter([idx2code[idx] for sprs_m in seq_data for idx in sprs_m.indices]).most_common()[:topK]

    logger.info("\n  TopK of [{}] codes: ".format(len(c)))
    for idx, r in enumerate(c): 
        try: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, code2title[r[0]]))
        except: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, '???'))
            
def acc_codeFrequency(logger, DATA_PATH, seq_data, thr):
    logger.info("\n  <acc_codeFrequency>\n")
    code2title = loadingFiles(logger, DATA_PATH, 'code2title.pkl')
    code2idx = loadingFiles(logger, DATA_PATH, 'code2idx.pkl')
    idx2code = {v:k for k,v in code2idx.items()}
    
    if type(seq_data)!=list:
        seq_data = [seq_data]
    from collections import Counter
    c = Counter([idx2code[idx] for sprs_m in seq_data for idx in sprs_m.indices]).most_common()       
    logger.info("\n  Top accumulated freq ({}) of codes: ".format(thr))
    count_all = sum([freq for _, freq in c])
    count_acc = 0
    for idx, r in enumerate(c): 
        if count_acc/count_all >= thr: 
            logger.info("\tstop at {0}".format(count_acc/count_all))
            break
        try: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, code2title[r[0]]))
        except: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, '???'))
        count_acc += r[1]
        
def Get_datasets_info(logger, DATASETS, dataset_type, cohort_type, topK=15, thr=0.1):
    """
    RESULT_FOLDER_PATH: where to log. /RESULT_FOLDER_PATH/report_datasets_info.log
    DATASETS: a dataset object
    dataset_type: TRAIN, TEST, or ALL
    cohort_type: TARGET, COMP, or ALL
    topK: to report topK-frequent codes
    thr: to report frequent codes (where accumulated sum of freq. <= thr)
    """
    
    import datetime
    logger.info("\n{}".format(datetime.datetime.now()))
    
    logger.info("\n(REPORT) {}-{}".format(dataset_type, cohort_type))
    for ds_type, dataset in zip(['TRAIN', 'TEST'], [DATASETS.train, DATASETS.test]):
        if ds_type!=dataset_type: 
            if dataset_type!='ALL':
                continue
        logger.info('\n[{0}-{1}]'.format(ds_type, cohort_type))
        if cohort_type=='TARGET':
            seq_data = dataset._t_ds._seq_data
            demo_data = dataset._t_ds._demo_data
            seq_len = dataset._t_ds._seq_len
        elif cohort_type=='COMP':
            seq_data = dataset._c_ds._seq_data
            demo_data = dataset._c_ds._demo_data
            seq_len = dataset._c_ds._seq_len
        elif cohort_type=='ALL':
            seq_data = dataset._t_ds._seq_data + dataset._c_ds._seq_data
            demo_data = dataset._t_ds._demo_data + dataset._c_ds._demo_data
            seq_len = dataset._t_ds._seq_len + dataset._c_ds._seq_len

        statsInfo(logger, seq_data, demo_data, seq_len)
        topK_codeFrequency(logger, DATASETS.info['DATA_FOLDER_PATH'], seq_data, topK)
        acc_codeFrequency(logger, DATASETS.info['DATA_FOLDER_PATH'], seq_data, thr)
    logger.info("\n  [DONE]\n\n")
    
def Stop_tensorboard():
    ## turn-off tensorboard service
    proc_list = get_ipython().getoutput(cmd='ps -ax | grep tensorboard')
    try:
        pid = [p.strip().split(' ')[0] for p in proc_list if 'grep' not in p][0]
        get_ipython().system_raw('kill -9 {}'.format(pid))
    
        import time
        time.sleep(1)
        print("[Stop] Tensorboard..\n")
    except:
        pass
    
def Run_tensorboard(LOGDIR):
    def _get_ip_address():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

    try:
        proc_list = get_ipython().getoutput(cmd='ps -ax | grep tensorboard')
    except:
        import subprocess
        proc_list = subprocess.check_output('ps -ax | grep tensorboard', 
                                            shell=True, universal_newlines=True).split('\n')

    IS_RUNNING = sum([1 if 'grep' not in p else 0 for p in proc_list]) > 0
    if IS_RUNNING:
        Stop_tensorboard()
    
    print("[Run] Tensorboard..\n")
    try:
        get_ipython().system_raw('tensorboard --logdir={} &'.format(LOGDIR))
    except:
        subprocess.call('tensorboard --logdir={} &'.format(LOGDIR), shell=True)
        
    import time
    time.sleep(1)
    try:
        IPADDRESS = _get_ip_address()
        if IPADDRESS.startswith('192') or IPADDRESS.startswith('10') or IPADDRESS.startswith('172'):
            IPADDRESS = 'localhost'
    except:
        IPADDRESS = 'localhost'
    print("[GO] http://{}:6006".format(IPADDRESS), "\n\n")
    
