# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:22:08 2019

@author: 1449486
"""

# getting the CORPUS
from pyexcel_ods import get_data
doc = get_data("corpus.ods")

c=doc['Sheet1']

data=[x[0] for x in c]

#......................splitting the data into TRAIN and TEST set..............

doc1 = get_data("hhh.ods")

import pandas as pd
c2=doc1['Sheet1']
#df=pd.DataFrame(c2)
tt=[x for x in c2 if x]
yy=[l.pop(2) for l in tt]

train_data=tt


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data, yy, test_size=0.33)

train1=[x[0] for x in x_train]
train2=[x[1] for x in x_train]

test1=[x[0] for x in x_test]
test2=[x[1] for x in x_test]

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#...............................................................................


import re

regex = '\(.{1,4}\)'

X=[]
for i in range(0,len(data)):
    x=data[i]
    x=x.replace('-',' ')
    x=x.replace('‘‘',' ')
    x=x.replace('"',' ')
    x=x.replace('‘‘',' ')
    x=re.sub(regex, ' ', x)
    x=x.replace('\n',' ') 
    x=x.replace('  ','') 
    x=x.strip()
    X.append(x)
      
import string

from nltk.tokenize import word_tokenize


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
#lmtzr.lemmatize('cars')

mm=[]
for i in data:
    tokens=word_tokenize(i)
    tokens=[w.lower() for w in tokens]
    table=str.maketrans('','',string.punctuation)
    stripped=[w.translate(table) for w in tokens]
    words=[word for word in stripped if word.isalpha()]
    filtered_sentence = [w for w in words if not w in stop_words]
    xx=[lmtzr.lemmatize(w) for w in filtered_sentence]
    mm.append(xx)


#..................TRAINING WORD EMBEDDINGS USING FASTTEXT..............................


EMBEDDING_DIM=50

from gensim.models import FastText
model_f = FastText(mm, size=EMBEDDING_DIM, window=5, min_count=1, workers=4,sg=1)

words=list(model_f.wv.vocab)

#.......................saving the embeddings matrix................................

filename='tracing_netword_embdd.txt'
model_f.wv.save_word2vec_format(filename,binary=False)


#.......................getting back the embeddings.........................
import numpy as np

import os
embeddings_index={}
f=open(os.path.join('','tracing_netword_embdd.txt'),encoding="utf-8")
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()



from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


  
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(mm)
sequences=tokenizer_obj.texts_to_sequences(mm)
  
word_index=tokenizer_obj.word_index
    
num_words=len(word_index)+1    
embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
max_length=max([len(s) for s in mm])


for word,i in word_index.items():
    if i>num_words:
        continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
e=embedding_matrix[1:len(embedding_matrix)]


#...............................................................................

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers import merge
from keras.models import Sequential, Model
from keras.layers import Concatenate,Flatten,Reshape,Dense, LSTM, Input, concatenate
import keras.backend as K
import keras
from keras.layers import multiply
from keras.layers import Input, dot, subtract, multiply,add,average
from keras.layers import Bidirectional
from keras.layers.core import Lambda
import tensorflow as tf


input_sentence_length=50
input_dim = EMBEDDING_DIM
num_steps = 100


#......................Input data conversion to padding.........................
#take data from mm..................................



X_train1_tokens=tokenizer_obj.texts_to_sequences(train1)
X_train2_tokens=tokenizer_obj.texts_to_sequences(train2)


X_train1_pad=pad_sequences(X_train1_tokens,maxlen=input_sentence_length,padding='post')

X_train2_pad=pad_sequences(X_train2_tokens,maxlen=input_sentence_length,padding='post')


X_test1_tokens=tokenizer_obj.texts_to_sequences(test1)
X_test2_tokens=tokenizer_obj.texts_to_sequences(test2)


X_test1_pad=pad_sequences(X_test1_tokens,maxlen=input_sentence_length,padding='post')

X_test2_pad=pad_sequences(X_test2_tokens,maxlen=input_sentence_length,padding='post')


#............................THE TRACING NEWORK................................
                      
main_input= Input(shape =(input_sentence_length,), dtype = 'int32', name = 'main_input')
aux_input = Input(shape =(input_sentence_length,), dtype = 'int32', name = 'aux_input')


embedding_layer1=Embedding(num_words,EMBEDDING_DIM,
                          embeddings_initializer=Constant(e),
                          input_length=input_sentence_length,
                          trainable=True)(main_input)

embedding_layer2=Embedding(num_words,EMBEDDING_DIM,
                          embeddings_initializer=Constant(e),
                          input_length=input_sentence_length,
                          trainable=True)(aux_input)


model1=Sequential()
lstm1=Bidirectional(LSTM(64,dropout=0.5,recurrent_dropout=0.5, 
                         input_shape=(num_steps, input_dim)))(embedding_layer1)

model1=Dense(EMBEDDING_DIM)(lstm1)

model2=Sequential()
lstm2=Bidirectional(LSTM(32,dropout=0.5, recurrent_dropout=0.5,
                         input_shape=(num_steps, input_dim)))(embedding_layer2)

model2=Dense(EMBEDDING_DIM)(lstm2)

direction=dot([model1, model2],axes=-1)

sub=subtract([model1, model2])

mult=multiply([sub,sub])

distance=Sequential()
distance=Dense(1,activation='relu',kernel_initializer=keras.initializers.Ones(),
                bias_initializer='zeros',trainable=False)(mult)


concat = concatenate([direction,distance])

mm=Sequential()
mm=Dense(1,activation='sigmoid')(concat)

model = Model(inputs=[main_input,aux_input], outputs=mm)

adam=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, 
                           decay=0, amsgrad=False)

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])


#................................MODEL SUMMARY..................................

print(model.summary())


#................................MODEL FITTING.................................
h=model.fit([X_train1_pad,X_train2_pad],y_train,batch_size=64,epochs=20,
            verbose=1,validation_data=([X_test1_pad,X_test2_pad],y_test))


#..........................MODEL SAVING AND LOADING.............................

from keras.models import load_model

model.save('tracing_network.h5')


model_loaded = load_model('tracing_network.h5')

#.................................PREDICTIONS...................................

import pandas as pd
predictions = model.predict([X_test1_pad,X_test2_pad])
from array import *
yy=predictions.tolist()
import itertools
y_pred=list(itertools.chain(*yy))
y_pred=np.array(y_pred)
import numpy as np
pred1 = np.where(y_pred > 0.5, 1, 0)
pred1=pd.DataFrame(pred1)

act=np.array(y_test)

actual=pd.DataFrame(act)

result=pd.concat([actual,pred1],axis=1)
result=pd.DataFrame(result.values, columns = ["Actual", "Predicted"])

#..................................CHECKING DATA..................................

x_t=pd.DataFrame(x_test)
check=pd.concat([x_t,result],axis=1)

check.to_csv('check.csv')


#................................CONFUSION MATRIX....................................

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred1)

print(cm)

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

print('---------------------------------------------')
print('Precision of the MODEL: ',precision(0,cm))
print('---------------------------------------------')
print('Recall    of the MODEL: ',recall(0,cm))
print('---------------------------------------------')
print('Accuracy  of the MODEL: ',accuracy(cm))
print('---------------------------------------------')



#..............................model plots..............................

import matplotlib.pyplot as plt


# summarize history for accuracy
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('model BiLSTM=32')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['Training_set', 'Validation_set'], loc='upper left')
plt.show()
