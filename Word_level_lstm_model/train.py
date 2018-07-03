
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os 
import data
import pickle
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import Model
import random
from metrics import kappa
import argparse
import logging


# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def training_loop(opts,training_set, batch_size, num_epochs, model, loss_, optim, training_iter, dev_iter, test_iter):
    step = 0
    epoch = 0
    total_batches = int(len(training_set) / batch_size)
    total_samples = total_batches * batch_size
    hidden = model.init_hidden(batch_size)
    while epoch <= num_epochs:
        epoch_loss = 0
        model.train()

        vectors, labels = get_batch(next(training_iter)) 
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        labels = Variable(torch.stack(labels).squeeze().type('torch.FloatTensor'))
        if opts.use_cuda:
            labels = labels.cuda() 
        vectors = Variable(vectors)
        if opts.use_cuda:
            vectors = vectors.cuda()

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(vectors, hidden)
        lossy = loss_(output, labels)
        epoch_loss += lossy.data[0] * batch_size

        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()

        best_dev = 0
        best_test = 0
        best_epoch = 0
        if step % total_batches == 0:
            kappa_dev = evaluate_kappa(opts, model, dev_iter,batch_size)
            kappa_test = evaluate_kappa(opts, model, test_iter,batch_size)
            # update best result
            if kappa_dev > best_dev:
                best_epoch = epoch
                best_dev = kappa_dev
                best_test = kappa_test
            with open(opts.path+"Bi-LSTM-prompt%d.txt"%opts.prompt, "a") as myfile:
                myfile.write("Epoch %i; Step %i; Dev kappa: %f; Test kappa: %f, best@ epoch%d: dev_kappa:%.3f, test_kappa:%.3f\n" 
                  %(epoch, step, kappa_dev, kappa_test,best_epoch, best_dev, best_test))
            print("Epoch %i; Step %i; Dev kappa: %f; Test kappa: %f, best@ epoch%d: dev_kappa:%.3f, test_kappa:%.3f\n" 
                  %(epoch, step, kappa_dev, kappa_test,best_epoch, best_dev, best_test))
            epoch += 1
            
        if step % 5 == 0:
            print("Epoch %i; Step %i; loss %f" %(epoch, step, lossy.data[0]))
        step += 1


def evaluate_kappa(opts, model, data_iter, batch_size):
    model.eval()
    predicted_labels = []
    true_labels = []
    hidden = model.init_hidden(batch_size)
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)

        if opts.use_cuda:
            vectors = vectors.cuda()
        vectors = Variable(vectors)
        
        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)

        predicted = [int(round(float(num))) for num in output.data.cpu().numpy()]
        predicted_labels.extend([round(float(num)) for num in output.data.cpu().numpy()])
        labels = [int(label[0]) for label in labels]
        true_labels.extend(labels)

    return kappa(true_labels, predicted_labels, weights = "quadratic")



def main():
    parser = argparse.ArgumentParser(description="sentence Hi_LSTM-attent-pooling model")
    parser.add_argument('--prompt',type = int, default = 1)
    parser.add_argument('--use_cuda',type = int, default = 0)
    parser.add_argument('--path', type=str, default = '../result/')
    args = parser.parse_args()

    # load data
    raw_data = pd.read_csv("../data/training_final_orig.csv", sep=',',header=0, index_col=0)
    raw_data = raw_data[raw_data.essay_set == args.prompt]
    data_set = data.get_data(raw_data)
    print('Finished Loading!')

    #get max sequence length
    max_seq_length = max(list(map(lambda x:len(x.split()),raw_data.essay)))
    print('max seq length: ', max_seq_length)

    # split to train/val/test
    data_size = len(data_set)
    print('prompt %d data_size: %d'%(args.prompt, data_size))
    training_set = data_set[:int(data_size*0.7)]
    dev_set = data_set[int(data_size*0.7):int(data_size*0.9)]
    test_set = data_set[int(data_size*0.9):]

    # convert and formatting
    word_to_ix, index_to_word, vocab_size = data.build_dictionary([training_set])
    print('vocab size', vocab_size)
    data.sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set, test_set], max_seq_length)
    print('Finished Converting!')


    # Hyper Parameters 
    model = 'LSTM'
    input_size = vocab_size
    hidden_dim = 100
    embedding_dim = 50
    batch_size = 10
    learning_rate = 0.001
    num_epochs = 50
    num_layer = 1
    bi_direction = True


    ################################################
    # pretrain loading     
    # filter GloVe, only keep embeddings in the vocabulary
    matrix = np.zeros((2, int(embedding_dim)))
    glove = {}
    filtered_glove = {}
    glove_path = '../data/filtered_glove_50.p'

    # reuse pickle file
    if(os.path.isfile(glove_path)):
        print("Reusing glove dictionary to save time")
        pretrained_embedding = pickle.load(open(glove_path,'rb'))
    else:
        with open('../data/glove.6B.50d.txt') as f:
            lines = f.readlines()
            for l in lines:
                vec = l.split(' ')
                glove[vec[0].lower()] = np.array(vec[1:])
        print('glove size={}'.format(len(glove)))
        print("Finished making glove dictionary")
        # search in vocabulary
        for i in range(2, len(index_to_word)):
            word = index_to_word[i]
            if(word in glove):
                vec = glove[word]
                filtered_glove[word] = glove[word]
                matrix = np.vstack((matrix,vec))
            else:
                # Random initialize
                random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,embedding_dim))
                matrix = np.vstack((matrix,random_init))

        pickle.dump(matrix, open("../data/filtered_glove_50.p", "wb"))
        pretrained_embedding = matrix
        print("Saving glove vectors")


    # Build, initialize, and train model
    rnn = Model.LSTM(model, vocab_size, embedding_dim, hidden_dim, num_layer, dropout=0.2, bidirectional=bi_direction)
    rnn.init_weights(args, pretrained_embedding)
    if args.use_cuda:
        rnn.cuda()

    # Loss and Optimizer
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the model
    training_iter = data_iter(training_set, batch_size)
    dev_iter = eval_iter(dev_set, batch_size)
    test_iter = eval_iter(test_set, batch_size)
          
    print('start training:')
    training_loop(args, training_set, batch_size, num_epochs, rnn, loss, optimizer, training_iter, dev_iter,test_iter)

if __name__ == '__main__':
    main()

