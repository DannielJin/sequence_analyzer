
import numpy as np
import tensorflow as tf
from . import dl_ops as ops
from .utils import get_logger_instance, loadingFiles, dumpingFiles

def param_dict_to_flag_grid(param_dict):
    #EMB_MATRIX_FILENAME
    if True in param_dict['EMB_MATRIX_FILENAME']:
        param_dict['EMB_MATRIX_FILENAME'] = [v for v in param_dict['EMB_MATRIX_FILENAME'] if v!=True] #remove True
        emb_filepath_list = [p.split('/')[-1] for p in glob.glob(param_dict['DATASETS'].info['DATA_FOLDER_PATH']+'*emb*')]
        param_dict['EMB_MATRIX_FILENAME'] += emb_filepath_list #add emb_filepath_list
    
    #make flag_grid
    flag_grid = dict()
    arch_num_set = {int(k.split('_')[2]) for k in param_dict.keys() if 'RNN_ARCH_' in k}
    
    flag_grid['RNN_ARCH'] = [dict() for _ in range(len(arch_num_set))]
    for k, v in param_dict.items():
        if k in ['DATASETS', 'NEW_GAME', 'CONFIG_FOLDER_PATH', 'RESULT_FOLDER_PATH', 'DUMPING_PATH',
                 'PROJECT_NAME', 'SA_PARAMS_FILE_NAME', 'DATASETS', 'NEW_GAME', 'TEST_ONLY', 'MODEL_INDICES']:
            continue
        if 'RNN_ARCH_' in k:
            arch_num_idx = int(k.split('_')[2])-1
            opt_name = '_'.join(k.split('_')[3:])
            if opt_name=='cell_type':
                v = v[0]
            flag_grid['RNN_ARCH'][arch_num_idx][opt_name] = v
        else:
            flag_grid[k] = v
    return flag_grid

def get_flag_list(flag_grid, DATA_FOLDER_PATH, MAX_TIME_STEP, FEATURE_SIZE, LABEL_SIZE, DUMPING_PATH):   
    #update flag_grid with datasets_info
    flag_grid['MAX_TIME_STEP'] = [MAX_TIME_STEP]
    flag_grid['FEATURE_SIZE'] = [FEATURE_SIZE]
    flag_grid['LABEL_SIZE'] = [LABEL_SIZE]
    flag_grid['DUMPING_PATH'] = [DUMPING_PATH]
    
    #flag_grid to flag_list
    from itertools import product
    flag_list = [dict(list(zip(list(flag_grid.keys()), values))) 
                 for values in list(product(*flag_grid.values()))]
    
    #check_validation
    flag_list_new = []
    for flag in flag_list:
        if 'EMB_MATRIX_FILENAME' in flag.keys():
            if flag['EMB_MATRIX_FILENAME']!=False:
                emb_matrix_shape = flag['EMB_MATRIX_FILENAME'].split('.pkl')[0].split('_')[-2:]

                #val_conditions
                cond1 = int(emb_matrix_shape[0])==flag['FEATURE_SIZE']
                cond2 = int(emb_matrix_shape[1])==flag['EMB_SIZE']
                if not cond1:
                    flag['EMB_MATRIX_FILENAME'] = False
                elif not cond2:
                    flag['EMB_MATRIX_FILENAME'] = False
        else: 
            flag['EMB_MATRIX_FILENAME'] = False
            
        if flag['MODEL_ARCH']=='RNN_MODEL':
            del flag['ATT_H_SIZE']
        
        flag_list_new.append(flag)
    
    #remove duplicated flag
    flag_list_new_unique = []
    for d_item in flag_list_new:
        if d_item not in flag_list_new_unique:
            flag_list_new_unique.append(d_item)
            
    #load emb_matrix && assign model_name
    for m_idx, flag in enumerate(flag_list_new_unique):
        try:
            if flag['EMB_MATRIX_FILENAME']==False:
                flag['EMB_MATRIX'] = False
            else:
                flag['EMB_MATRIX'] = loadingFiles(None, DATA_FOLDER_PATH, flag['EMB_MATRIX_FILENAME'], verbose=False)
        except:
            flag['EMB_MATRIX'] = False
            
        flag['MODEL_NAME'] = 'MODEL_{}'.format(m_idx+1)
        
    #dumping
    dumpingFiles(None, DUMPING_PATH, 'flag_list.pkl', flag_list_new_unique)
    
    return flag_list_new_unique
            

def Get_model_list(param_dict, DUMPING_PATH, model_indices=[-1]):
    """
    param_dict; if False(or None), /DUMPING_PATH/flag_list.pkl will be loaded.
    model_indices; List of model_num which you want to build. If model_indices==[-1], all models will be used.
    """
    if (param_dict) or (param_dict is not None):
        ##make flag_list
        flag_grid = param_dict_to_flag_grid(param_dict)
        flag_list = get_flag_list(flag_grid, 
                                  param_dict['DATASETS'].info['DATA_FOLDER_PATH'], 
                                  param_dict['DATASETS'].info['MAX_TIME_STEP'], 
                                  param_dict['DATASETS'].info['FEATURE_SIZE'], 
                                  param_dict['DATASETS'].info['LABEL_SIZE'],
                                  param_dict['DUMPING_PATH'])
    else:
        ##load flag_list.pkl
        flag_list = loadingFiles(None, DUMPING_PATH, 'flag_list.pkl')
    
    ##get model_list
    MODEL_DICT = {'RNN_MODEL': RNN_MODEL, 
                  'RNN_ATTENTION_MODEL': RNN_ATTENTION_MODEL, 
                  'BIRNN_ATTENTION_MODEL': BIRNN_ATTENTION_MODEL}
    
    if model_indices==[-1]:
        model_indices = ['MODEL_{}'.format(m_idx+1) for m_idx in range(len(flag_list))]
    else:
        model_indices = ['MODEL_{}'.format(m_num) for m_num in model_indices]
    
    model_list = []
    for m_idx, flag in enumerate(flag_list):
        if flag['MODEL_NAME'] in model_indices:
            model_list.append(MODEL_DICT[flag['MODEL_ARCH']](flag))
    return model_list


class RNN_MODEL():
    def __init__(self, flag):
        self.flag = flag
        self.tensorDict = dict()
        self.resultDict = dict()
        self.g = tf.Graph()
        self.g_vis = tf.Graph()
        self.Building_graph()
        
    def _get_logger(self):
        from .utils import get_logger_instance
        self.logger = get_logger_instance(logger_name=self.flag['MODEL_NAME'], 
                                          DUMPING_PATH=self.flag['DUMPING_PATH'])
        
    def _basic_tensors(self):
        with tf.name_scope('Learning_Rate'):    
            self.tensorDict['global_step'] = tf.Variable(0, name="Global_step", trainable=False, dtype=tf.int32)
            if ('DECAY_STEPS' in self.flag.keys())&('DECAY_RATE' in self.flag.keys()):
                self.tensorDict['lr'] = tf.train.exponential_decay(self.flag['LR'], self.tensorDict['global_step'], 
                                                               self.flag['DECAY_STEPS'], self.flag['DECAY_RATE'], 
                                                               staircase=True, name='ExpDecay_lr')
            else:
                self.tensorDict['lr'] = tf.constant(self.flag['LR'], name='Constant_lr')
    
    def _input_layer_tensors(self):
        with tf.name_scope('Input_Layer'):
            self.tensorDict['inputs'] = tf.placeholder(tf.float32, 
                                                       shape=[self.flag['BATCH_SIZE'], 
                                                              self.flag['MAX_TIME_STEP'], 
                                                              self.flag['FEATURE_SIZE']], 
                                                       name='Inputs')
            self.tensorDict['labels'] = tf.placeholder(tf.float32, 
                                                       shape=[self.flag['BATCH_SIZE'], 
                                                              self.flag['LABEL_SIZE']], 
                                                       name='Labels')
            self.tensorDict['demo'] = tf.placeholder(tf.float32, 
                                                     shape=[self.flag['BATCH_SIZE'], 
                                                            self.flag['MAX_TIME_STEP'], 
                                                            2], 
                                                     name='Demo')
            self.tensorDict['seq_lens'] = tf.placeholder(tf.int32, 
                                                         shape=[self.flag['BATCH_SIZE']], 
                                                         name='Seq_lens')
            self.tensorDict['keep_prob'] = tf.placeholder(tf.float32, shape=[], name='Keep_prob')
            

    def _embedding_layer_tensors(self):
        with tf.name_scope('EMB_Layer'):
            if self.flag['EMB_MATRIX'] is not False:
                self.tensorDict['W_emb'] = tf.constant(self.flag['EMB_MATRIX'], tf.float32, name='W_emb')
                self.tensorDict['emb_inputs'] = tf.tensordot(self.tensorDict['inputs'], self.tensorDict['W_emb'], 
                                                             axes=1, name='Emb_inputs')
            else:
                self.tensorDict['W_emb'] = tf.Variable(tf.random_normal(shape=[self.flag['FEATURE_SIZE'], 
                                                                               self.flag['EMB_SIZE']], 
                                                                        stddev=0.04), 
                                                       name='W_emb')
                self.tensorDict['emb_inputs'] = tf.nn.sigmoid(tf.tensordot(self.tensorDict['inputs'], 
                                                                           self.tensorDict['W_emb'], 
                                                                           axes=1), 
                                                              name='Emb_inputs')
                
            self.tensorDict['emb_and_demo'] = tf.concat([self.tensorDict['emb_inputs'], self.tensorDict['demo']], 
                                                        axis=-1, name='emb_and_demo')
    
    def _RNN_layer_tensors(self):
        # Inputs: self.tensorDict['emb_inputs']
        with tf.name_scope('RNN_Layer'):
            cell_list = []
            for i in range(len(self.flag['RNN_ARCH']['hidden_size'])):
                cell = ops.translator(self.flag['RNN_ARCH']['cell_type'])(self.flag['RNN_ARCH']['hidden_size'][i],
                                                                          activation=tf.nn.tanh,
                                                                          name='RNN_{}_{}'.format(self.flag['RNN_ARCH']['cell_type'], i+1))
                if self.flag['RNN_ARCH']['drop_out'][i]:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.tensorDict['keep_prob'])
                cell_list.append(cell)
            self.tensorDict['RNN_cells'] = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            self.tensorDict['RNN_outputs'], self.tensorDict['RNN_states'] = tf.nn.dynamic_rnn(self.tensorDict['RNN_cells'], 
                                                                                              self.tensorDict['emb_and_demo'], 
                                                                                              self.tensorDict['seq_lens'], 
                                                                                              dtype=tf.float32)
            #self.tensorDict['RNN_outputs_last'] = tf.ruduce_sum(self.tensorDict['RNN_outputs'], axis=1)
            self.tensorDict['RNN_outputs_last'] = ops.get_last_output(self.tensorDict['RNN_outputs'],
                                                                      self.tensorDict['seq_lens'],
                                                                      name='RNN_outputs_last')
            
    def _prediction_layer_tensors(self):    
        with tf.variable_scope('Pred_Layer'):
            self.tensorDict['W_pred'] = tf.Variable(tf.random_normal(shape=[self.flag['RNN_ARCH']['hidden_size'][-1], 
                                                                            self.flag['LABEL_SIZE']], 
                                                                     stddev=0.04), 
                                                    name='W_pred')
            self.tensorDict['b_pred'] = tf.Variable(0.01, name='b_pred')
            
            self.tensorDict['logits'] = tf.add(tf.matmul(self.tensorDict['RNN_outputs_last'], self.tensorDict['W_pred']), 
                                               self.tensorDict['b_pred'],
                                               name='Logits')
            self.tensorDict['pred'] = tf.nn.softmax(self.tensorDict['logits'], axis=-1, name='pred')
            
    def _Inference(self):        
        with tf.name_scope('Inference'):
            self._embedding_layer_tensors()
            self._RNN_layer_tensors()
            self._prediction_layer_tensors()
        
    def _Loss(self):
        with tf.variable_scope('Loss'):
            xentropy_mean = tf.losses.softmax_cross_entropy(onehot_labels=self.tensorDict['labels'],
                                                            logits=self.tensorDict['logits'], 
                                                            weights=1.0,
                                                            scope='xentropy_mean')
            
            xentropy_mean_weighted = tf.losses.softmax_cross_entropy(onehot_labels=self.tensorDict['labels'],
                                                                     logits=self.tensorDict['logits'], 
                                                                     weights=self.flag['CLASS_WEIGHT'],
                                                                     scope='xentropy_mean_weighted')
            
            if self.flag['EMB_MATRIX'] is not False:
                loss_reg = tf.add_n([tf.nn.l2_loss(self.tensorDict['W_pred']), 
                                     tf.nn.l2_loss(self.tensorDict['b_pred'])],
                                    name='loss_l2reg')
            else:
                loss_reg = tf.add_n([tf.nn.l2_loss(self.tensorDict['W_pred']), 
                                     tf.nn.l2_loss(self.tensorDict['b_pred']),
                                     tf.nn.l2_loss(self.tensorDict['W_emb'])],
                                    name='loss_l2reg') 
            self.tensorDict['loss'] = tf.add(xentropy_mean, self.flag['L2_REG']*loss_reg, name='loss')
            self.tensorDict['loss_weighted'] = tf.add(xentropy_mean_weighted, self.flag['L2_REG']*loss_reg, name='loss_weighted')
            
    def _Optimizer(self):
        with tf.name_scope('Optimizer'):       
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.tensorDict['loss_weighted'], params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5, name='CLIP_GRAD')
            optimizer = tf.train.AdamOptimizer(self.tensorDict['lr'])
            self.tensorDict['trainOp'] = optimizer.apply_gradients(zip(clipped_gradients, params), 
                                                                   global_step=self.tensorDict['global_step'],
                                                                   name='UPDATE')
            
    def _Evaluation(self):
        with tf.name_scope('Evaluation'):
            self.tensorDict['accuracy'] = ops.accuracy(self.tensorDict['pred'], self.tensorDict['labels'])
            
    def _Summary(self):
        ## logging
        self.logger.info("\n[FLAG]")
        for k, v in self.flag.items():
            self.logger.info("\t{}:  {}".format(k, v))
            
        self.logger.info("\n[INPUT_LAYERS]")
        self.logger.info("\tinputs: {}".format(self.tensorDict['inputs']))
        self.logger.info("\tlabels: {}".format(self.tensorDict['labels']))
        self.logger.info("\tdemo: {}".format(self.tensorDict['demo']))
        self.logger.info("\tseq_lens: {}".format(self.tensorDict['seq_lens']))
        self.logger.info("\tkeep_prob: {}".format(self.tensorDict['keep_prob']))
        
        self.logger.info("\n[EMB_LAYERS]")
        self.logger.info("\tW_emb: {}".format(self.tensorDict['W_emb']))
        self.logger.info("\temb_inputs: {}".format(self.tensorDict['emb_inputs']))
        self.logger.info("\temb_and_demo: {}".format(self.tensorDict['emb_and_demo']))
                    
        self.logger.info("\n[RNN_LAYERS]")
        self.logger.info("\tRNN_cells: {}".format(self.tensorDict['RNN_cells']))
        self.logger.info("\tRNN_outputs: {}".format(self.tensorDict['RNN_outputs']))
        self.logger.info("\tRNN_outputs_last: {}".format(self.tensorDict['RNN_outputs_last']))
        
        self.logger.info("\n[PREDICTION_LAYERS]")
        self.logger.info("\tW_pred: {}".format(self.tensorDict['W_pred']))
        self.logger.info("\tb_pred: {}".format(self.tensorDict['b_pred']))
        self.logger.info("\tlogits: {}".format(self.tensorDict['logits']))
        self.logger.info("\tpred: {}".format(self.tensorDict['pred']))
        
        self.logger.info("\n[LOSS]")
        self.logger.info("\tloss: {}".format(self.tensorDict['loss']))
        self.logger.info("\tloss_weighted: {}".format(self.tensorDict['loss_weighted']))
        
        self.logger.info("\n[ACCUARCY]")
        self.logger.info("\taccuracy: {}".format(self.tensorDict['accuracy']))
        
        tf.summary.scalar('lr', self.tensorDict['lr'])
        tf.summary.scalar('loss', self.tensorDict['loss'])
        tf.summary.scalar('loss_weighted', self.tensorDict['loss_weighted'])
        tf.summary.scalar('accuracy', self.tensorDict['accuracy'])
        tf.summary.histogram('W_emb', self.tensorDict['W_emb'])
        tf.summary.histogram('W_pred', self.tensorDict['W_pred'])
        tf.summary.histogram('RNN_outputs_last', self.tensorDict['RNN_outputs_last'])
        tf.summary.histogram('logits', self.tensorDict['logits'])
        tf.summary.histogram('pred', self.tensorDict['pred'])
        self.tensorDict['summary_op'] = tf.summary.merge_all()
            
    def Building_graph(self):
        with self.g.as_default():
            self._basic_tensors()
            self._input_layer_tensors()
            self._Inference()
            self._Loss()
            self._Optimizer()
            self._Evaluation()
            self._get_logger()
            self._Summary()
            
            
class RNN_ATTENTION_MODEL(RNN_MODEL):    
    def _RNN_layer_tensors(self):
        with tf.name_scope('RNN_Layer'):
            cell_list = []
            for i in range(len(self.flag['RNN_ARCH']['hidden_size'])):
                cell = ops.translator(self.flag['RNN_ARCH']['cell_type'])(self.flag['RNN_ARCH']['hidden_size'][i],
                                                                          activation=tf.nn.tanh,
                                                                          name='RNN_{}_{}'.format(self.flag['RNN_ARCH']['cell_type'], i+1))
                if self.flag['RNN_ARCH']['drop_out'][i]:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.tensorDict['keep_prob'])
                cell_list.append(cell)
            self.tensorDict['RNN_cells'] = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            self.tensorDict['RNN_outputs'], self.tensorDict['RNN_states'] = tf.nn.dynamic_rnn(self.tensorDict['RNN_cells'], 
                                                                                              self.tensorDict['emb_and_demo'], 
                                                                                              self.tensorDict['seq_lens'], 
                                                                                              dtype=tf.float32)
            
    def _attention_layer_tensors(self):
        with tf.name_scope('Attention_Layer'):
            self.tensorDict['W1_att'] = tf.Variable(tf.random_normal(shape=[self.flag['RNN_ARCH']['hidden_size'][-1], 
                                                                            self.flag['ATT_H_SIZE']], 
                                                                     stddev=0.04), 
                                                    name='W1_att')
            self.tensorDict['b1_att'] = tf.Variable(tf.random_normal(shape=[self.flag['ATT_H_SIZE']], stddev=0.04), 
                                                    name='b1_att')
            self.tensorDict['W2_att'] = tf.Variable(tf.random_normal(shape=[self.flag['ATT_H_SIZE']], stddev=0.04), 
                                                    name='W2_att')
            
            self.tensorDict['att_hidden'] = tf.tanh(tf.tensordot(self.tensorDict['RNN_outputs'], self.tensorDict['W1_att'], 
                                                                 axes=1) + self.tensorDict['b1_att'], 
                                                    name='att_hidden')
            self.tensorDict['att_alpha'] = tf.nn.softmax(tf.tensordot(self.tensorDict['att_hidden'], self.tensorDict['W2_att'],
                                                                      axes=1),
                                                         name='att_alpha')
            self.tensorDict['att_outputs'] = tf.reduce_sum(tf.multiply(self.tensorDict['RNN_outputs'],
                                                                       tf.expand_dims(self.tensorDict['att_alpha'], -1)), 
                                                           axis=1, name='att_outputs')
                        
    def _prediction_layer_tensors(self):    
        with tf.variable_scope('Pred_Layer'):
            self.tensorDict['W_pred'] = tf.Variable(tf.random_normal(shape=[self.flag['RNN_ARCH']['hidden_size'][-1], 
                                                                            self.flag['LABEL_SIZE']], 
                                                                     stddev=0.04), 
                                                    name='W_pred')
            self.tensorDict['b_pred'] = tf.Variable(0.01, name='b_pred')
            
            self.tensorDict['logits'] = tf.add(tf.matmul(self.tensorDict['att_outputs'], self.tensorDict['W_pred']), 
                                               self.tensorDict['b_pred'],
                                               name='Logits')
            self.tensorDict['pred'] = tf.nn.softmax(self.tensorDict['logits'], axis=-1, name='pred')
                        
    def _Inference(self):        
        with tf.name_scope('Inference'):
            super()._embedding_layer_tensors()
            self._RNN_layer_tensors()
            self._attention_layer_tensors()
            self._prediction_layer_tensors()
            
    def _Summary(self):
        ## logging
        self.logger.info("\n[FLAG]")
        for k, v in self.flag.items():
            self.logger.info("\t{}:  {}".format(k, v))
            
        self.logger.info("\n[INPUT_LAYERS]")
        self.logger.info("\tinputs: {}".format(self.tensorDict['inputs']))
        self.logger.info("\tlabels: {}".format(self.tensorDict['labels']))
        self.logger.info("\tdemo: {}".format(self.tensorDict['demo']))
        self.logger.info("\tseq_lens: {}".format(self.tensorDict['seq_lens']))
        self.logger.info("\tkeep_prob: {}".format(self.tensorDict['keep_prob']))
        
        self.logger.info("\n[EMB_LAYERS]")
        self.logger.info("\tW_emb: {}".format(self.tensorDict['W_emb']))
        self.logger.info("\temb_inputs: {}".format(self.tensorDict['emb_inputs']))
        self.logger.info("\temb_and_demo: {}".format(self.tensorDict['emb_and_demo']))
                    
        self.logger.info("\n[RNN_LAYERS]")
        self.logger.info("\tRNN_cells: {}".format(self.tensorDict['RNN_cells']))
        self.logger.info("\tRNN_outputs: {}".format(self.tensorDict['RNN_outputs']))
        
        self.logger.info("\n[ATTENTION_LAYERS]")
        self.logger.info("\tW1_att: {}".format(self.tensorDict['W1_att']))
        self.logger.info("\tb1_att: {}".format(self.tensorDict['b1_att']))
        self.logger.info("\tW2_att: {}".format(self.tensorDict['W2_att']))
        self.logger.info("\tatt_hidden: {}".format(self.tensorDict['att_hidden']))
        self.logger.info("\tatt_alpha: {}".format(self.tensorDict['att_alpha']))
        self.logger.info("\tatt_outputs: {}".format(self.tensorDict['att_outputs']))

        self.logger.info("\n[PREDICTION_LAYERS]")
        self.logger.info("\tW_pred: {}".format(self.tensorDict['W_pred']))
        self.logger.info("\tb_pred: {}".format(self.tensorDict['b_pred']))
        self.logger.info("\tlogits: {}".format(self.tensorDict['logits']))
        self.logger.info("\tpred: {}".format(self.tensorDict['pred']))
        
        self.logger.info("\n[LOSS]")
        self.logger.info("\tloss: {}".format(self.tensorDict['loss']))
        self.logger.info("\tloss_weighted: {}".format(self.tensorDict['loss_weighted']))
        
        self.logger.info("\n[ACCUARCY]")
        self.logger.info("\taccuracy: {}".format(self.tensorDict['accuracy']))

        ## summary
        tf.summary.scalar('lr', self.tensorDict['lr'])
        tf.summary.scalar('loss', self.tensorDict['loss'])
        tf.summary.scalar('loss_weighted', self.tensorDict['loss_weighted'])
        tf.summary.scalar('accuracy', self.tensorDict['accuracy'])
        tf.summary.histogram('att_hidden', self.tensorDict['att_hidden'])
        tf.summary.histogram('att_alpha', self.tensorDict['att_alpha'])
        tf.summary.histogram('att_outputs', self.tensorDict['att_outputs'])
        tf.summary.histogram('W_emb', self.tensorDict['W_emb'])
        tf.summary.histogram('W_pred', self.tensorDict['W_pred'])
        tf.summary.histogram('logits', self.tensorDict['logits'])
        tf.summary.histogram('pred', self.tensorDict['pred'])
        self.tensorDict['summary_op'] = tf.summary.merge_all()
            
            
class BIRNN_ATTENTION_MODEL(RNN_ATTENTION_MODEL):              
    def _RNN_layer_tensors(self):
        with tf.name_scope('RNN_Layer'):
            fw_cell_list = []
            bw_cell_list = []
            for i in range(len(self.flag['RNN_ARCH']['hidden_size'])):
                fw_cell = ops.translator(self.flag['RNN_ARCH']['cell_type'])(self.flag['RNN_ARCH']['hidden_size'][i]/2,
                                                                             activation=tf.nn.tanh,
                                                                             name='RNN_fw_{}_{}'.format(self.flag['RNN_ARCH']['cell_type'], i+1))
                bw_cell = ops.translator(self.flag['RNN_ARCH']['cell_type'])(self.flag['RNN_ARCH']['hidden_size'][i]/2,
                                                                             activation=tf.nn.tanh,
                                                                             name='RNN_bw_{}_{}'.format(self.flag['RNN_ARCH']['cell_type'], i+1))
                if self.flag['RNN_ARCH']['drop_out'][i]:
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, self.tensorDict['keep_prob'])
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, self.tensorDict['keep_prob'])
                fw_cell_list.append(fw_cell)
                bw_cell_list.append(bw_cell)
                
            self.tensorDict['RNN_cells'] = (tf.nn.rnn_cell.MultiRNNCell(fw_cell_list), tf.nn.rnn_cell.MultiRNNCell(bw_cell_list))
            biRNN_outputs, self.tensorDict['RNN_states'] = tf.nn.bidirectional_dynamic_rnn(self.tensorDict['RNN_cells'][0], 
                                                                                           self.tensorDict['RNN_cells'][1], 
                                                                                           self.tensorDict['emb_and_demo'], 
                                                                                           self.tensorDict['seq_lens'], 
                                                                                           dtype=tf.float32)
            self.tensorDict['RNN_outputs'] = tf.concat(biRNN_outputs, axis=2)
            
    def _Inference(self):        
        with tf.name_scope('Inference'):
            super()._embedding_layer_tensors()
            self._RNN_layer_tensors()
            super()._attention_layer_tensors()
            super()._prediction_layer_tensors()
            
            
            