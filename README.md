# Automated-Scoring-System
Automated Scoring System - DS-1011 Natural Language Processing

See AES Result.pdf for final result.

## Data
* ** https://www.kaggle.com/c/asap-aes**

* 8 topics(prompts):
    * 1783 essays in Topic 1.
    * 1800 essays in Topic 2.
    * 1726 essays in Topic 3.
    * 1771 essays in Topic 4.
    * 1805 essays in Topic 5.
    * 1800 essays in Topic 6.
    * 1569 essays in Topic 7.
    * 723 essays in Topic 8.


* Score Ranges:
    * Class 1 range 2- 12
    * Class 2 range 1- 6
    * Class 3 range 0- 3
    * Class 4 range 0- 3
    * Class 5 range 0- 4
    * Class 6 range 0- 4
    * Class 7 range 0- 30
    * Class 8 range 0- 60
    
    
## Preprocessing

* Punctuation cleaning
* Name/Location indexer replace
* tokenize sentences to words with nltk.word_tokenize
* tokenize essays to sentences
* Normalize scores to 0-1


## Model

#### Linear Regression(Baseline):
Extract 16 features listed below, also applied 5-fold cross validation and forward feature selection to find the best feature settings.
    1. word count 
    2. long word count
    3. noun word count
    4. verb count
    5. comma count
    6. punctuation count
    7. sentence count
    8. adjective count
    9. adverb count
    10. lexical diversity
    11. quatation mark
    12. word length
    13. spelling error
    14*.bracket count
    15*.exclamation count
    16*. Foreign words count


#### Word Level Bi-directional LSTM:
* Use longest essay's length, 0 padding shorter essay
* Use Pretrained GloVe, leave fine-tuned in the training process
* Mean over Time pooling on hidden states
* Train by topics


#### Stacked Model
Stack sentence level model on word level model. First use a lstm/cnn layer on word level, then apply a attention pooling/mean over time pooling layer to study the sentence representation, then use another lstm/cnn layer on each sentence representation, finally apply a attention pooling/mean over time pooling layer to study the final essay reprentation, which is used to make score prediction.

> Example: CNN-MoT-CNN-Mot model
_________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    word_input (InputLayer)      (None, 3550)              0         
    _________________________________________________________________
    x (Embedding)                (None, 3550, 50)          200000    
    _________________________________________________________________
    drop_x (Dropout)             (None, 3550, 50)          0         
    _________________________________________________________________
    resh_W (Reshape)             (None, 71, 50, 50)        0         
    _________________________________________________________________
    z (TimeDistributed)          (None, 71, 46, 100)       25100     
    _________________________________________________________________
    avg_z (TimeDistributed)      (None, 71, 1, 100)        0         
    _________________________________________________________________
    resh_z (Reshape)             (None, 71, 100)           0         
    _________________________________________________________________
    hz (Conv1D)                  (None, 69, 100)           30100     
    _________________________________________________________________
    avg_hz (GlobalAveragePooling (None, 100)               0         
    _________________________________________________________________
    output (Dense)               (None, 1)                 101       
    =================================================================
    Total params: 255,301.0
    Trainable params: 255,301.0
    Non-trainable params: 0.0


## Metrics:
* QWK: quatratic weighted kappa

QWK measure the consistancy between two series of scores.

## Result
* Linear regression: 0.69 (mean of 8 topics, below same)
* Word level Bi-directional LSTM: 0.71
* Stacked model: LSTM-att-LSTM-att: 0.78 (best)
