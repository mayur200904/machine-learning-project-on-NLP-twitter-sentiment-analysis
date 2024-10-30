#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:35:24 2024

@author: mayursantoshtarate
"""

import numpy as np
import pandas as pd


#load the dataset
data=pd.read_csv('Twitter_Data.csv') 

# clean text columns 
data['selected_text']=data['text'].astype(str)
data['text']=data['selected_text'].str.replace('[^a-zA-Z\\s]',' ').str.lower()

# split the data in X and Y
X=data['text']
y=data['sentiment']

# check the unique values of sentiment 
unique_sentiment =y.unique()
print("Unique Sentiments:",unique_sentiment)

# convert the sentiments into label
y=y.replace({'negative':0,'neutral':1,'positive':2})

# divide data into training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# tokenize and pad text sequence
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer= Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq=tokenizer.texts_to_sequences(X_train)
X_test_seq=tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq,maxlen=100,padding='post')
X_test_pad = pad_sequences(X_test_seq,maxlen=100,padding='post')

# encode labels 
from sklearn.preprocessing import LabelEncoder 
labelencoder=LabelEncoder()
y_train_encoded=labelencoder.fit_transform(y_train)
y_test_encoded=labelencoder.transform(y_test)

# one hot encode of labels- tensorflow 
num_classes=len(unique_sentiment)
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded,num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded,num_classes=num_classes)

# model building 
from tensorflow.keras.layers import LSTM,Embedding,Dense,Dropout

# build lstm model with 3 output units
model= tf.keras.Sequential(
    [
     Embedding(input_dim=5000,output_dim=100,input_length=100),
     LSTM(128),
     Dense(64,activation='relu'),
     Dropout(0.5),
     Dense(num_classes,activation='softmax')
     ])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# train my model
model.fit(X_train_pad,y_train_onehot,epochs=5,batch_size=32,validation_data=(X_test_pad,y_test_onehot))
model.summary()

# save the model
from joblib import dump

model.save('sentiment_model.h5')
dump(tokenizer,'tokenizer.joblib')
