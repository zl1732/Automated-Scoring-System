# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-09 16:54:09
# @Last Modified by:   zl1732
# @Last Modified time: 2018-07-03 21:36:02

import os
import sys
import argparse
import random
import numpy as np
from utils import *
import hier_lstm
from keras.callbacks import ModelCheckpoint
from data_prepare import prepare_sentence_data
from evaluator import Evaluator
import time


logger = get_logger("Train sentence sequences sents-HiLSTM")
np.random.seed(100)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_LSTM-attent-pooling model")
    parser.add_argument('--model',choices=['build_model', 'build_bidirectional_model', 'build_attention_model', 'build_attention2_model'],required = True)
    
    parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove or random')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
    
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')


    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--l2_value', type=float, default=0.001, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory')

    parser.add_argument('--train',default = '../data/train.tsv')  # "data/word-level/*.train"
    parser.add_argument('--dev',default = '../data/dev.tsv')
    parser.add_argument('--test',default = '../data/test.tsv')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')

    # to support init bias of output layer of the network
    parser.add_argument('--init_bias', action='store_true', help='initial bias of output' + 
                        'layer with the mean score of training data')

    # parser
    args = parser.parse_args()
    model = args.model
    checkpoint_dir = args.checkpoint_path
    
    # model name
    modelname = "%s.prompt%s.%sunits.bs%s.hdf5" % (model, args.prompt_id, args.lstm_units, args.batch_size)
    modelpath = os.path.join(checkpoint_dir, modelname)
    
    
    # load data
    datapaths = [args.train]#, args.dev, args.test]
    embedding_path = args.embedding_dict
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    prompt_id = args.prompt_id
    (X_t, Y_t, mask_train), vocab, vocab_size, embed_table, max_sentlen, max_sentnum, init_mean_value = \
        prepare_sentence_data(datapaths, '../data/vocab_essay_set%d.pk'%args.prompt_id,\
            embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, \
            to_lower=True, sort_by_len=False)

        
    # load pretrained embedding
    if embed_table is not None:
        embedd_dim = embed_table.shape[1]
        embed_table = [embed_table]

    nn1 = int(np.ceil(len(X_t)*0.7))
    nn2 = int(np.ceil(len(X_t)*0.9))

    Y_train = Y_t[0:nn1]
    Y_dev = Y_t[nn1:nn2]
    Y_test = Y_t[nn2:]

    X_train = X_t.reshape((X_t.shape[0], X_t.shape[1]*X_t.shape[2]))[0:nn1]
    X_dev = X_t.reshape((X_t.shape[0], X_t.shape[1]*X_t.shape[2]))[nn1:nn2]
    X_test = X_t.reshape((X_t.shape[0], X_t.shape[1]*X_t.shape[2]))[nn2:]
  

    # create log
    fh = logging.FileHandler('%s.log'%modelpath)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # BUild model
    model = getattr(hier_lstm, model)(args, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)
    logger.info("X_train shape: %s" % str(X_train.shape))

    # Evaluation
    evl = Evaluator(args.prompt_id, False, checkpoint_dir, modelname, X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
    
    # Initial evaluation
    logger.info("Initial evaluation: ")
    evl.evaluate(model, -1, logger, print_info=True)
    logger.info("Train model")
    for ii in range(args.num_epochs):
        logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
        start_time = time.time()
        history =  model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evl.evaluate(model, ii+1,logger)
        evl.print_info(logger)


if __name__ == '__main__':
    main()

