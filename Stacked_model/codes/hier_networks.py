# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:40:53
# @Last Modified by:   zl1732
# @Last Modified time: 2018-07-03 21:36:02

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, merge
from keras.layers import TimeDistributed

from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import l2

from softattention import Attention
from zeromasking import ZeroMaskedEntries
from utils import get_logger
import time

logger = get_logger("Build model")

"""
Hierarchical networks, the following function contains several models:
(1)build_hcnn_model: hierarchical CNN model
(2)build_hrcnn_model: hierarchical Recurrent CNN model, LSTM stack over CNN,
 it supports two pooling methods
    (a): Mean-over-time pooling
    (b): attention pooling
"""


def build_hcnn_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False):

    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, filter2_len = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.nbfilters, opts.filter1_len, opts.filter2_len, opts.dropout, opts.l2_value))

    """
    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)
    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)
    z = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='z')(resh_W)
    avg_z = TimeDistributed(AveragePooling1D(pool_length=L-opts.filter1_len+1), name='avg_z')(z)	# shape= (N, 1, nbfilters)
    resh_z = Reshape((N, opts.nbfilters), name='resh_z')(avg_z)		# shape(N, nbfilters)
    hz = Convolution1D(opts.nbfilters, opts.filter2_len, border_mode='valid', name='hz')(resh_z)
    # avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)
    avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz)
    model = Model(input=word_input, output=y)
    """
    model = Sequential()
    model.add(Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights,name='x'))
    model.add(Dropout(opts.dropout, name='drop_x'))
    model.add(Reshape((N, L, embedd_dim), dtype='int32', input_shape=(N*L,), name='resh_W'))
    model.add(TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='z'))
    model.add(TimeDistributed(AveragePooling1D(pool_length=L-opts.filter1_len+1), name='avg_z'))
    model.add(Reshape((N, opts.nbfilters),name='resh_z'))
    model.add(Convolution1D(opts.nbfilters, opts.filter2_len, border_mode='valid', name='hz'))
    model.add(GlobalAveragePooling1D(name='avg_hz'))
    model.add(Dense(output_dim=1, activation='sigmoid', name='output')) 
    
    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model


def build_hrcnn_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    # LSTM stacked over CNN based on sentence level
    N = maxnum
    L = maxlen
    print(opts)
    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    
    # embedding layer
    if opts.use_mask == 0:
        x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=False, name='x')(word_input)
        x_maskedout = x
        
    elif opts.use_mask == 1:
        x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
        x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)

    # drop out
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)
    # reshape
    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)
    # CNN layer
    zcnn = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='zcnn')(resh_W)

    # pooling mode1 on CNN
    if opts.mode1 == 'mot':
        logger.info("Use mean-over-time pooling on sentence")
        avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
    elif opts.mode1 == 'att':
        logger.info('Use attention-pooling on sentence')
        avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
    elif opts.mode1 == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on sentence')
        avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), input_shape = (K.int_shape(zcnn)[2],K.int_shape(zcnn)[3]),name='avg_zcnn1')(zcnn)
        avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2')(zcnn)
        avg_zcnn = merge([avg_zcnn1, avg_zcnn2], mode='concat', name='avg_zcnn')
    else:
        raise NotImplementedError


    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    # pooling mode1 on LSTM
    if opts.mode2 == 'mot':
        logger.info('Use mean-over-time pooling on text')
        avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode2 == 'att':
        logger.info('Use attention-pooling on text')
        avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode2 == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on text')
        avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1')(hz_lstm)
        avg_hz_lstm2 = Attention(name='avg_hz_lstm2')(hz_lstm)
        avg_hz_lstm = merge([avg_hz_lstm1, avg_hz_lstm2], mode='concat', name='avg_hz_lstm')
    else:
        raise NotImplementedError

    # l2 regularization
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(output_dim=1, activation='sigmoid', name='output', kernel_regularizer=regularizers.l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz_lstm)


    model = Model(input=word_input, output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].bias = bias_value

    if verbose:
        model.summary()


    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model


