
import tensorflow as tf
import numpy as np

def matmul_with_reshape(x, w, name, b=False):
    multiplied = tf.reshape(tf.matmul(tf.reshape(x, [-1, x.shape[-1]]), w), 
                            [-1, x.shape[1], w.shape[-1]], name=name)
    if b: multiplied = tf.add(multiplied, b, name=name)
    return multiplied

def masking(x, seq_lens, time_size, name):
    if len(x.shape)==3:
        mask = tf.tile(tf.expand_dims(tf.sequence_mask(seq_lens, time_size), axis=2), 
                       [1, 1, x.shape[-1]], name='Mask')
    else:
        mask = tf.sequence_mask(seq_lens, time_size, name='Mask')
    return tf.where(mask, x, tf.zeros_like(x), name=name)

def get_last_output(x, seq_lens, name):
    idx = tf.stack([tf.range(tf.shape(x)[0]), seq_lens-1], axis=1, name='Last_time')
    return tf.gather_nd(x, idx, name=name)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def accuracy(pred, true):
    is_correct = tf.equal(tf.argmax(pred,1), tf.argmax(true,1))
    return tf.reduce_mean(tf.cast(is_correct, tf.float32), name='Accuracy')

def translator(arg):
    tf_dict = {'relu': tf.nn.relu, 'lrelu': lrelu, 'sigmoid':tf.nn.sigmoid, 'tanh':tf.nn.tanh, 
               'LSTM':tf.nn.rnn_cell.LSTMCell, 'GRU':tf.nn.rnn_cell.GRUCell}
    try: return tf_dict[arg]
    except: return arg


def CONV_LAYERS(inputs, conv_arch, keep_prob):    
    print("\n[CONV_LAYERS]")
    
    try:
        if conv_arch['conv1d']:
            conv_opt = tf.layers.conv1d
            pool_opt = tf.layers.max_pooling1d
            pooling_str = 2
    except:
        conv_opt = tf.layers.conv2d
        pool_opt = tf.layers.max_pooling2d
        pooling_str = [2,2]
        
    conved = inputs
    for i in range(len(conv_arch['hidden_size'])):
        conv_input = conved
        conv_input = tf.layers.batch_normalization(conv_input)
        conved = conv_opt(
                            conv_input,
                            filters=conv_arch['hidden_size'][i],
                            kernel_size=conv_arch['k_size'][i],
                            activation=translator(conv_arch['activation'][i]),
                            strides=conv_arch['stride'][i],
                            padding="same",
                            name="conv2d_%d" % i)
        
        if conv_arch['pooling'][i]:
            conved = pool_opt(conved, 
                              pool_size=conv_arch['pooling'][i],
                              strides=pooling_str,
                              padding="same",
                              name="pool_%d" % i)
                              
        if conv_arch['drop_out'][i]:
            conved = tf.nn.dropout(conved, keep_prob, name='conv_dropOut_%d'%i)
        print("\t", conved)
    return conved


def RNN_LAYERS(inputs, rnn_arch, keep_prob, seqLens):    
    print("\n[RNN_LAYERS]")
    cell_list = []
    for i in range(len(rnn_arch['hidden_size'])):
        cell = translator(rnn_arch['cell_type'])(rnn_arch['hidden_size'][i])
        if rnn_arch['drop_out'][i]:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, keep_prob)
        cell_list.append(cell)
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cell_list)
    print(rnn_cells)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cells, inputs, sequence_length=seqLens, dtype=tf.float32)
    print("\tRNN_outputs:", rnn_outputs)
        
    mask = tf.tile(tf.expand_dims(tf.sequence_mask(seqLens, tf.shape(rnn_outputs)[1]), axis=2), 
                   [1,1,tf.shape(rnn_outputs)[2]])
    return tf.where(mask, rnn_outputs, tf.zeros_like(rnn_outputs))

def FC_LAYERS(inputs, fc_arch, keep_prob):        
    print("\n[FC_LAYERS]")
    fc = inputs
    for i in range(len(fc_arch['hidden_size'])):
        fc_input = fc
        fc = tf.layers.dense(
                            fc,
                            units=fc_arch['hidden_size'][i],
                            activation=translator(fc_arch['activation'][i]),
                            name="fc_%d" % i)
        
        if fc_arch['drop_out'][i]:
            fc = tf.nn.dropout(fc, keep_prob, name='fc_dropOut_%d'%i)
        print("\t", fc)
    return fc

