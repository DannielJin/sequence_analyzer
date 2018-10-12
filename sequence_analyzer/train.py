
import tensorflow as tf
import numpy as np
from .utils import get_logger_instance, loadingFiles, dumpingFiles, option_printer
from .report import draw_roc_curve, draw_pr_curve, get_group_indices, get_attention_codes

def Train_model(logger, dataSets, model, logPath, new_game=True):
    import os
    
    with tf.Session(graph=model.g) as sess:
        tf.global_variables_initializer().run(session=sess)
        tf.train.start_queue_runners(sess=sess)
        tr_writer = tf.summary.FileWriter(logPath+'/train', graph=sess.graph)
        val_writer = tf.summary.FileWriter(logPath+'/val', graph=sess.graph)
        #summary_op = tf.summary.merge_all()
        
        ## LOAD
        if not new_game: loading = load_model(logger, logPath, sess)
        else: 
            if logger is not None:
                logger.info("[@] New game..")
            else:
                print("[@] New game..")
        
        if logger is not None: 
            logger.info("\ntraining..\n")
        else:
            print("\ntraining..\n")
            
        for i in range(model.flag['TRAIN_STEPS']):
            batch_train_data = dataSets.train.next_batch(model.flag['BATCH_SIZE'])
            batch_train_pid_list = batch_train_data[0]
            tr_feed_dict = {model.tensorDict['inputs']: batch_train_data[1], 
                            model.tensorDict['labels']: batch_train_data[2], 
                            model.tensorDict['demo']: batch_train_data[3], 
                            model.tensorDict['seq_lens']: batch_train_data[4],
                            model.tensorDict['keep_prob']: model.flag['KEEP_PROB']}
            _ = sess.run(model.tensorDict['trainOp'], feed_dict=tr_feed_dict)
            
            if ((i+1)%model.flag['PRINT_BY'])==0:
                g_step, lr = sess.run([model.tensorDict['global_step'], 
                                       model.tensorDict['lr']], 
                                      feed_dict=tr_feed_dict)
                ## from train_data
                tr_loss, tr_acc = sess.run([model.tensorDict['loss'],
                                            model.tensorDict['accuracy']], 
                                           feed_dict=tr_feed_dict)
                tr_summary = sess.run(model.tensorDict['summary_op'], feed_dict=tr_feed_dict)
                tr_writer.add_summary(tr_summary, g_step)
                tr_writer.flush()
                
                ## from val_data
                batch_val_data = dataSets.test.next_batch(model.flag['BATCH_SIZE'])
                batch_val_pid_list = batch_val_data[0]
                val_feed_dict = {model.tensorDict['inputs']: batch_val_data[1], 
                                 model.tensorDict['labels']: batch_val_data[2], 
                                 model.tensorDict['demo']: batch_val_data[3], 
                                 model.tensorDict['seq_lens']: batch_val_data[4],
                                 model.tensorDict['keep_prob']: 1.0}
                val_loss, val_acc = sess.run([model.tensorDict['loss'],
                                              model.tensorDict['accuracy']], 
                                             feed_dict=val_feed_dict)
                val_summary = sess.run(model.tensorDict['summary_op'], feed_dict=val_feed_dict)
                val_writer.add_summary(val_summary, g_step)
                val_writer.flush()
                
                if logger is not None:
                    logger.info('[G-{}/{}]  LOSS ({:.4f}/{:.4f})  ACC ({:.2f}/{:.2f}) LR ({:6f})'
                                .format(g_step, model.flag['TRAIN_STEPS'], tr_loss, val_loss, tr_acc, val_acc, lr))
                else:
                    print('[G-{}/{}]  LOSS ({:.4f}/{:.4f})  ACC ({:.2f}/{:.2f}) LR ({:6f})'
                          .format(g_step, model.flag['TRAIN_STEPS'], tr_loss, val_loss, tr_acc, val_acc, lr))
            
            if (i!=0)&(((i+1)%model.flag['SAVE_BY'])==0):
                save_model(logger, logPath, sess, g_step=model.tensorDict['global_step'])
                
        ## SAVE
        save_model(logger, logPath, sess, g_step=model.tensorDict['global_step'])
        
        if logger is not None:
            logger.info("\n[DONE@@]")
        else:
            print("\n[DONE@@]")
        
            
## JIN_add_at_180829        
def Test_model(logger, datasets, model, logPath):
    import numpy as np
    import os
    
    pid_list = datasets.test._t_ds._pid_list + datasets.test._c_ds._pid_list
    inputs = datasets.test._t_ds._seq_data + datasets.test._c_ds._seq_data
    labels = datasets.test._t_ds._labels + datasets.test._c_ds._labels
    demo = datasets.test._t_ds._demo_data + datasets.test._c_ds._demo_data 
    labels = datasets.test._t_ds._labels + datasets.test._c_ds._labels 
    seq_lens = datasets.test._t_ds._seq_len + datasets.test._c_ds._seq_len 
    n_examples = len(inputs)
    if logger is not None:
        logger.info("\ntesting..\n")
        logger.info("  test_examples: {}".format(n_examples))
    else:
        print("\ntesting..\n")
        print("  test_examples: {}".format(n_examples))
    
    batch_size = model.flag['BATCH_SIZE']
    
    with tf.Session(graph=model.g) as sess:
        tf.global_variables_initializer().run(session=sess)
        loading = load_model(logger, logPath, sess)
        if not loading:
            if logger is not None:
                logger.info("\n  [ABORT !!]")
            else:
                print("\n  [ABORT !!]")
            return [None, None, None, None, None, None]
            
        pred_all = []
        true_all = []
        if model.flag['MODEL_ARCH']=='RNN_ATTENTION_MODEL':
            att_code_all = []
            #make idx2title
            code2idx = loadingFiles(logger, datasets.info['DATA_FOLDER_PATH'], 'code2idx.pkl')
            code2title = loadingFiles(logger, datasets.info['DATA_FOLDER_PATH'], 'code2title.pkl')
            idx2title = {v:code2title[k] for k,v in code2idx.items()}
        loss_all = 0
        acc_all = 0
        index_in_epoch = 0        
        for i in range(n_examples//batch_size):
            start = index_in_epoch
            index_in_epoch += batch_size
            end = index_in_epoch            
            test_feed_dict = {model.tensorDict['inputs']: [sprs_m.toarray() for sprs_m in inputs[start:end]], 
                              model.tensorDict['labels']: labels[start:end], 
                              model.tensorDict['demo']: demo[start:end], 
                              model.tensorDict['seq_lens']: seq_lens[start:end],
                              model.tensorDict['keep_prob']: 1.0}
            pred, true, test_loss, test_acc = sess.run([model.tensorDict['pred'], 
                                                        model.tensorDict['labels'],
                                                        model.tensorDict['loss'],
                                                        model.tensorDict['accuracy']],
                                                       feed_dict=test_feed_dict)            
            pred_all.append(pred)
            true_all.append(true)
            loss_all += test_loss
            acc_all += test_acc
            
            ##attention part
            if model.flag['MODEL_ARCH']=='RNN_ATTENTION_MODEL':
                alphas = sess.run(model.tensorDict['att_alpha'], feed_dict=test_feed_dict)
                batch_inputs = np.array([sprs_m.toarray() for sprs_m in inputs[start:end]])
                batch_labels = np.array(labels[start:end])
                                
                #get att_codes
                g_true_positive, g_true_negative, g_false_positive, g_false_negative = get_group_indices(pred, batch_labels)
                tp_att_codes = get_attention_codes(batch_inputs[g_true_positive], alphas[g_true_positive], idx2title, 
                                                   topK_time=3, topK_code=15)
                #tn_att_codes = get_attention_codes(batch_inputs[g_true_negative], alphas[g_true_negative], idx2title, 
                #                                   topK_time=3, topK_code=15)
                #fp_att_codes = get_attention_codes(batch_inputs[g_false_positive], alphas[g_false_positive], idx2title, 
                #                                   topK_time=3, topK_code=15)
                #fn_att_codes = get_attention_codes(batch_inputs[g_false_negative], alphas[g_false_negative], idx2title, 
                #                                   topK_time=3, topK_code=15)
                att_code_all.append(tp_att_codes)
                
    pred_all = np.array([p for b_p in pred_all for p in b_p])
    true_all = np.array([t for b_t in true_all for t in b_t])
    avg_batch_loss = loss_all/max((n_examples//batch_size), 1)
    avg_batch_acc = acc_all/max((n_examples//batch_size), 1)
    
    if model.flag['MODEL_ARCH']=='RNN_ATTENTION_MODEL':
        att_code_all_dict = dict()
        for att_code_batch in att_code_all:
            for title, freq in att_code_batch:
                if title in att_code_all_dict.keys():
                    att_code_all_dict[title] += freq
                else:
                    att_code_all_dict[title] = freq
        att_code_all = [(k,v) for k,v in att_code_all_dict.items()]
        
        import pandas as pd
        df = pd.DataFrame(att_code_all, columns=['Code', 'Freq'])
        dumpingFiles(logger, '/'.join(logPath.split('/')[:-1]), 'df_ATT_RESULTS_{}.pkl'.format(logPath.split('/')[-1]), df)
        df.to_html(os.path.join('/'.join(logPath.split('/')[:-1]), 'df_ATT_RESULTS_{}.html'.format(logPath.split('/')[-1])))
        if logger is not None:
            logger.info("df_ATT_RESULTS.html dumped.. {}".format('/'.join(logPath.split('/')[:-1])))
        else:
            print("df_ATT_RESULTS.html dumped.. {}".format('/'.join(logPath.split('/')[:-1])))
        
    
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    auroc = roc_auc_score(true_all[:, 1], pred_all[:, 1])
    auprc = average_precision_score(true_all[:, 1], pred_all[:, 1])
    micro_f1_score = f1_score(np.argmax(true_all, axis=1), np.argmax(pred_all, axis=1), average='micro')
    weighted_f1_score = f1_score(np.argmax(true_all, axis=1), np.argmax(pred_all, axis=1), average='weighted')
    
    if logger is not None:
        logger.info("\n[Report]")
        logger.info("  Avg of batch_loss: {}\n  Avg of batch_acc: {}".format(avg_batch_loss, avg_batch_acc))
        logger.info("  AUROC of {}-samples: {}".format(n_examples, auroc))
        logger.info("  AUPRC of {}-samples: {}".format(n_examples, auprc))
        logger.info("  Micro_F1_Score of {}-samples: {}".format(n_examples, micro_f1_score))
        logger.info("  Weighted_F1_Score of {}-samples: {}".format(n_examples, weighted_f1_score))
    else:
        print("\n[Report]")
        print("  Avg of batch_loss: {}\n  Avg of batch_acc: {}".format(avg_batch_loss, avg_batch_acc))
        print("  AUROC of {}-samples: {}".format(n_examples, auroc))
        print("  AUPRC of {}-samples: {}".format(n_examples, auprc))
        print("  Micro_F1_Score of {}-samples: {}".format(n_examples, micro_f1_score))
        print("  Weighted_F1_Score of {}-samples: {}".format(n_examples, weighted_f1_score))
    
    draw_roc_curve(logPath, true_all, pred_all)
    draw_pr_curve(logPath, true_all, pred_all)
    
    return [avg_batch_loss, avg_batch_acc, auroc, auprc, micro_f1_score, weighted_f1_score]
    
    
def save_model(logger, logPath, sess, g_step):
    import os
    if not os.path.exists(logPath): os.makedirs(logPath)
    tf.train.Saver().save(sess, os.path.join(logPath, os.path.basename(logPath)), global_step=g_step)
    if logger is not None:
        logger.info(" [*] Saving checkpoints... {}".format(logPath))
    else:
        print(" [*] Saving checkpoints... {}".format(logPath))
        
def load_model(logger, logPath, sess):
    import os
    ckpt = tf.train.get_checkpoint_state(os.path.abspath(logPath))
    try:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        if logger is not None:
            logger.info("Loading SUCCESS.. ") 
        else:
            print("Loading SUCCESS.. ") 
        return True
    except: 
        if logger is not None:
            logger.info("Loading FAILED.. ")
        else:
            print("Loading FAILED.. ")
        return False
        

def Train_model_list(MODEL_LIST, DATASETS, DUMPING_PATH, new_game=False):
    """
    Train and Test models
    new_game: if new_game, /RESULT_BASE_PATH/PROJECT_NAME/DB_NAME will be FORMATTED. Initialize saved models and figures.
    """
    import os, datetime
    from .utils import get_logger_instance
    from .report import Run_tensorboard
        
    if not os.path.exists(DUMPING_PATH): 
        os.makedirs(DUMPING_PATH)
        
    #tensorboard
    Run_tensorboard(DUMPING_PATH)
    
    #clear DUMPING_PATH/*
    if new_game:
        import shutil, glob, os
        _ = [shutil.rmtree(p) for p in glob.glob(os.path.join(DUMPING_PATH, '**/'))] #remove MODEL_* folders
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*.pkl')) if 'flag_list.pkl' not in p]
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*.html'))]
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*.png'))]
        #_ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, 'MODEL_*.log'))]
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*_model_list.log'))]
    
    #logging
    logger = get_logger_instance(logger_name='train_model_list', 
                                 DUMPING_PATH=DUMPING_PATH, 
                                 parent_name='sa_pipeline', 
                                 stream=False)
    logger.info("\n{}".format(datetime.datetime.now()))
    if new_game:
        logger.info("\n(Previous Logs removed)\n")
    logger.info("[Train_model_list]\n")
    
    #train models
    for idx, model in enumerate(MODEL_LIST):
        logger.info("\n\t[@] MODEL-({}/{}) Training.. \n".format(idx+1, len(MODEL_LIST)))
        logger.info("  (model_params)")
        option_printer(logger, **model.flag)
        logPath = os.path.join(model.flag['DUMPING_PATH'], model.flag['MODEL_NAME'])

        # training
        Train_model(logger, DATASETS, model, logPath, new_game)
        
        # testing
        _ = Test_model(logger, DATASETS, model, logPath)
        
    logger.info("\n[ALL DONE]")
    
    
def Test_model_list(MODEL_LIST, DATASETS, DUMPING_PATH):
    """
    Test models and get model_results
    """
    import os, datetime
    from .utils import get_logger_instance
        
    logger = get_logger_instance(logger_name='test_model_list', 
                                 DUMPING_PATH=DUMPING_PATH, 
                                 parent_name='sa_pipeline', 
                                 stream=False)
    logger.info("\n{}".format(datetime.datetime.now()))
    logger.info("[Test_model_list]")
    
    RESULTS = []
    for idx, model in enumerate(MODEL_LIST):
        logger.info("\n\t[@] MODEL-({}/{}) Testing.. \n".format(idx+1, len(MODEL_LIST)))
        logger.info("  (model_params)")
        option_printer(logger, **model.flag)
        logPath = os.path.join(model.flag['DUMPING_PATH'], model.flag['MODEL_NAME'])
        
        # testing
        results = Test_model(logger, DATASETS, model, logPath)
        RESULTS.append([model.flag['MODEL_NAME']]+results+[model.flag])
    
    import pandas as pd
    df = pd.DataFrame(RESULTS, columns=['Model_Name', 'Avg_Batch_Loss', 'Avg_Batch_Acc', 'AUROC', 'AUPRC', 
                                        'Micro_F1_Score', 'Weighted_F1_Score', 'Flag'])
    dumpingFiles(logger, DUMPING_PATH, 'df_RESULTS.pkl', df)
    df.to_html(os.path.join(DUMPING_PATH, 'df_RESULTS.html'))
    logger.info("df_RESULTS.html dumped.. {}".format(DUMPING_PATH))
    logger.info("\n[ALL DONE]")



