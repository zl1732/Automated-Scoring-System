# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-07 16:01:25
# @Last Modified by:   zl1732
# @Last Modified time: 2018-07-03 21:36:02
import reader
import utils
import keras.backend as K
import numpy as np

logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(datapath, vocab_path, embedding_path=None, embedding='glove', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, to_lower=True, sort_by_len=False):

    assert len(datapath) == 1, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts), vocab, overal_maxlen, overal_maxnum = reader.get_data(datapath, vocab_path, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False)
        
        

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    
        
    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    
    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    #######################################
    #这里我改了   prompt_id ->train_prompts
    #######################################
    if prompt_id ==-1:
        Y_train = reader.get_model_friendly_scores(y_train, train_prompts)
        scaled_train_mean = Y_train.mean()
    elif prompt_id!=-1:
        Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
        scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, Y_train, mask_train), vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean



