#!/usr/bin/env python
# coding: utf-8

# ### LSTM Model for Sentiment Prediction

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer       
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import classification_report

import re

import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Activation, TimeDistributed, SimpleRNN, LSTM, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import random

from itertools import islice


# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
df = df.drop([1,2,3,4], axis=1)
df.columns = ['sentiment', 'text']


# In[4]:


random_idx_list = [510513,1496539,702286,1540390,559338]
df = df.loc[random_idx_list,:]
print(df.head())


# In[17]:


# reading the data
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
df = df.drop([1,2,3,4], axis=1)
df.columns = ['sentiment', 'text']

# try with a model with only 100,000 data points
random_idx_list = [random.randint(1,len(df.text)) for i in range(100000)] 
df = df.loc[random_idx_list,:]

# df now consists of only the text and associated sentiment
print(df.head())
print(df.shape)


# In[5]:


# 0 = negative, 1 = positive
df.sentiment = df.sentiment.apply(lambda x: 1 if (x == 4) else 0)


# In[17]:


pd.set_option('display.width', 3000000)
pd.set_option('display.max_colwidth', 130)
pd.options.display.max_rows = 4000
df


# In[7]:


# plotting data for visualization
val_count = df.sentiment.value_counts()
plt.figure(figsize=(4,4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")


# In[8]:


# text-preprocessing: removing hyperlinks, hashtags and twitter mentions
def cleanText(text):
    text = re.sub(r'https?://[^\s\n\r]+','',text.lower())
    text = re.sub(r'@[^\s\n\r]+','',text)
    text = re.sub(r'#','',text)
    
    # remove any unnecessary double spaces
    text = re.sub(r'  ',' ',text).strip()
    return text

df.text = df.text.apply(cleanText)
df.head()


# In[9]:


# tokenizing tweets
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
df.text = df.text.apply(tokenizer.tokenize)
df.head()


# In[10]:


# removing stop words and punctuations (cleaning the data up a bit more)
stopwords_english = stopwords.words('english')

def cleanStopWords(token):
    cleaned_token = []
    for word in token:
        if(word not in stopwords_english and word not in string.punctuation):
            cleaned_token.append(word)
    return cleaned_token

df.text = df.text.apply(cleanStopWords)
df.head()


# In[11]:


# stemming the words in the tweet tokens
stemmer = PorterStemmer()

def cleanStemWords(token):
    cleaned_token = []
    for word in token:
        cleaned_token.append(stemmer.stem(word))
    return cleaned_token

df.text = df.text.apply(cleanStemWords)
df.head()


# In[10]:


# returns a dictionary containing the frequency at which the words have appeared in the text
# input: list of tweet and the sentiments (1s and 0s)
# output: dictionary with positive and negative frequencies of the strings in the tweets
def build_freqs(data, label):
  freqs = {}
  for i in range(len(data)):
    for word in data.iat[i]:
      pair = (word, label.iat[i])
      if pair in freqs:
        freqs[pair] = freqs[pair] + 1
      else:
        freqs[pair] = 1    
  return freqs

frequencies = build_freqs(df.text, df.sentiment)
dict(islice(frequencies.items(), 0, 5))


# In[11]:


# NOTE: This function is only there for testing, and can be ignored. 
# prints the length of each of the wordfrequency
def printLength(wordFrequency, count):
    lengthList = []
    for minWordCount in range(count):
        counter = 0
        for word in wordFrequency:
            if(wordFrequency[word] > minWordCount):
                counter += 1
        lengthList.append(counter)
    return lengthList


# In[12]:


# creates a dictionary that shows the frequency with which each word has appeared in all the tweets
def createWordFrequency(frequencies):
    wordFrequency = {}
    for pair in frequencies:
        wordFrequency[pair[0]] = wordFrequency.get(pair[0],0) + frequencies[pair]
    return wordFrequency
    
# create a mapping from all vocab words to indices with the inputted frequency list and minimum word count
def wordToIndex(maxLen, wordFrequency):
    word_to_index = {}
    word_to_index["<START>"] = 1 # start tag
    word_to_index["<UNK>"] = 2 # unknown words
    
    # fill in the other vocab words
    counter = 2
    for word in wordFrequency:
        if wordFrequency[word] > maxLen:
            counter += 1
            word_to_index[word] = counter
            
    return word_to_index
    
wordFrequency = createWordFrequency(frequencies)
# create a word_to_index dictionary for words that appeared more than 3 times in all tweets.
word_to_index = wordToIndex(3, wordFrequency)
print(len(wordFrequency))
print(len(word_to_index))


# In[13]:


valSize = 0.20
# since the data is processed, separate the data into a training and validation set
trainX, valX, trainY, valY = train_test_split(df.text, df.sentiment.to_numpy(), test_size=valSize, random_state=4)
print(trainX.head())
print(trainY[0:5])


# In[14]:


# find the longest length of a tweet
def maxLength(tweetTokens):
    maximum = 0
    for token in tweetTokens:
        if(len(token) > maximum):
            maximum = len(token)
    return maximum

# maxLen is used as the input length in the embedding layer of the model
maxLen = maxLength(df.text)
print(maxLen)

# encode and pad all the words in a tweet to their indices.
def encode(tweetList, word_to_index):
    # encode
    encoded_data = []
    for tweet in tweetList:
        encoded_line = [1] # <start> tag
        for word in tweet:
            if word in word_to_index:
                encoded_line.append(word_to_index[word])
            else:
                encoded_line.append(2) # unknown words
        encoded_data.append(encoded_line)
    
    # pad
    padded_encoded_data = pad_sequences(encoded_data,maxlen=maxLen,truncating='post')
    
    return padded_encoded_data

x_train = encode(trainX, word_to_index)
x_val = encode(valX, word_to_index)
print(x_train[0])
print(x_train.shape)


# In[15]:


# hyper-parameters for the model
y_train = trainY
y_val = valY
EMBEDDING_DIM = 200
DROPOUT = 0.40
n1 = 16
n2 = 4
EPOCHS = 5
LR = 0.003


# In[16]:


# create LSTM model
def createModel(EMBEDDING_DIM,INPUT_LENGTH,N1,N2,DROPOUT, EPOCHS, LR, x_train, y_train, x_val, y_val): 
    # define constants
    VOCAB_LEN = len(word_to_index) - 2
    
    # create model 
    model = 0
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_LEN,output_dim=EMBEDDING_DIM,input_length=INPUT_LENGTH))
    model.add(LSTM(n1,dropout=DROPOUT))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(Adam(learning_rate=LR),loss='binary_crossentropy',metrics=['accuracy'])
    
    # train model
    model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val, y_val))
    
    return model
        
model = createModel(EMBEDDING_DIM, maxLen,n1, n2, DROPOUT, EPOCHS, LR, x_train, y_train, x_val, y_val) 


# In[17]:


# plot results
J_list = model.history.history['loss']
plt.plot(J_list)


# In[18]:


# print losses for training and validation dataset
bce = BinaryCrossentropy(from_logits=False)
y_train_hat = model.predict(x_train)
print("Training:")
print(bce(y_train.reshape(-1,1), y_train_hat).numpy())
print('---')
print("Validation:")
y_val_hat = model.predict(x_val)
print(bce(y_val.reshape(-1,1), y_val_hat).numpy())


# In[19]:


y_train_hat_cat = 1*(model.predict(x_train) > 0.5)
print(classification_report(y_train,y_train_hat_cat))


# In[20]:


y_val_hat_cat = 1*(model.predict(x_val) > 0.5)
print(classification_report(y_val,y_val_hat_cat))


# In[ ]:




