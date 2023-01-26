#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

import random


# In[20]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[21]:


# reading the data
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
df = df.drop([1,2,3,4], axis=1)
df.columns = ['sentiment', 'text']
# try with a smaller model
random_idx_list = [random.randint(1,len(df.text)) for i in range(100000)] 
df = df.loc[random_idx_list,:]
# df now consists of only the text and associated sentiment
print(df.head())
print(df.shape)


# In[22]:


# 0 = negative, 1 = positive
df.sentiment = df.sentiment.apply(lambda x: 1 if (x == 4) else 0)
df.tail()


# In[23]:


val_count = df.sentiment.value_counts()
plt.figure(figsize=(4,4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")


# In[24]:


# text-preprocessing: removing hyperlinks, hashtags and twitter mentions
def cleanText(text):
    text = re.sub(r'https?://[^\s\n\r]+','',text.lower())
    text = re.sub(r'@[^\s\n\r]+','',text)
    text = re.sub(r'#','',text)
    # remove any unnecessary double spaces
    text = re.sub(r'  ',' ',text).strip()
    return text
df.text = df.text.apply(cleanText)


# In[25]:


# tokenizing tweets
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
df.text = df.text.apply(tokenizer.tokenize)


# In[26]:


# removing stop words and punctuations (cleaning the data up a bit more)
stopwords_english = stopwords.words('english')

def cleanStopWords(token):
    cleaned_token = []
    for word in token:
        if(word not in stopwords_english and word not in string.punctuation):
            cleaned_token.append(word)
    return cleaned_token

df.text = df.text.apply(cleanStopWords)


# In[27]:


# stemming the words in the tweet tokens
stemmer = PorterStemmer()

def cleanStemWords(token):
    cleaned_token = []
    for word in token:
        cleaned_token.append(stemmer.stem(word))
    return cleaned_token

df.text = df.text.apply(cleanStemWords)


# In[28]:


# returns a dictionary containing the frequency at which the words have appeared in the text
# input: list of tweet and the sentiments (1s and 0s)
# outpu: dictionary with positive and negative frequencies of the strings in the tweets
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


# In[29]:


# returns the frequency with which the word has appeared among all the tweets for the inputted sentiment
def word_freq(freq,word,sentiment):
    if (word,sentiment) in freq:
        return freq[(word,sentiment)]
    return 0 

# returns an array of features for all the tweets. 
# First column contains the total number of times each word in the tweet has appeared in a negative comment.
# Second column contains the total number of times each word in the tweet has appeared in a positive comment.
def calculate_features(tweets,freqs):
 features = np.zeros((len(tweets),2))
 for i in range(len(tweets)):
   tweet = tweets.iat[i]
   for word in tweet:
     features[i,0] = features[i,0] + word_freq(freqs,word,0)
     features[i,1] = features[i,1] + word_freq(freqs,word,1)
 return features

# calculate the features based only on the training data.
features = calculate_features(df.text,frequencies)
features


# In[30]:


valSize = 0.20
# since the data is processed, separate the data into a training and validation set
trainX, valX, trainY, valY = train_test_split(features, df.sentiment.to_numpy(), test_size=valSize, random_state=4)


# In[31]:


# scale the data
scaler = preprocessing.StandardScaler()
scaler.fit(trainX)
X_train_scaled = scaler.transform(trainX)
X_val_scaled = scaler.transform(valX)


# In[32]:


# creating the model -- logistic regression
model = 0
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy')
model.fit(X_train_scaled,trainY,epochs=10,verbose=0)
J_list = model.history.history['loss']
plt.plot(J_list)


# In[33]:


def decision_boundary():
  xl, xr, dx = 0, 1000, 10
  yl, yr, dy = 0, 1000, 10
  u = np.arange(xl,xr,dx)
  v = np.arange(yl,yr,dy)
  u_r = np.ones((len(v),1))*u.reshape(1,len(u))
  v_r = v.reshape(len(v),1)*np.ones((1,len(u)))
  u_r = u_r.reshape(-1)
  v_r = v_r.reshape(-1)
  p_grid = np.column_stack((u_r,v_r))
  p_grid_scaled = scaler.transform(p_grid)
  f_grid = model.predict(p_grid_scaled)
  f_grid = f_grid.reshape((len(v),len(u)))
  plt.contour(u,v,f_grid,levels=[0.5])
  return


# In[34]:


# visualizing our features
sns.scatterplot(features[:,0],features[:,1],hue=df.sentiment)
decision_boundary()
plt.xlabel('negative tweet frequencies',fontsize=20)
plt.ylabel('positive tweet frequencies',fontsize=20)
plt.xlim(0,1000)
plt.ylim(0,1000)


# In[37]:


bce = BinaryCrossentropy(from_logits=False)
y_train_hat = model.predict(X_train_scaled)
print("Training:")
print(bce(trainY.reshape(-1,1), y_train_hat).numpy())
print('---')
print("Validation:")
y_val_hat = model.predict(X_val_scaled)
print(bce(valY.reshape(-1,1), y_val_hat).numpy())


# In[38]:


y_train_hat_cat = 1*(model.predict(X_train_scaled) > 0.5)
print(classification_report(trainY,y_train_hat_cat))


# In[39]:


y_val_hat_cat = 1*(model.predict(X_val_scaled) > 0.5)
print(classification_report(valY,y_val_hat_cat))


# In[75]:


# prints the performance results of the model with the inputted number of epochs, dense layers, 
# and the number of nodes in each of the dense layers
def tryModel(epochs, nodeList):
    EPOCH_INCREASE = 10
    denseNum = len(nodeList)
    
    # create a new model
    model = 0
    model = Sequential()
    for i in range(denseNum):
        # final layer should have activation as sigmoid otherwise relu function
        if(i == denseNum-1):
            model.add(Dense(nodeList[i], activation='sigmoid'))
        else:
            model.add(Dense(nodeList[i], activation='relu'))
    model.compile(loss='binary_crossentropy')
        
    for currentEpochs in range(10,epochs,EPOCH_INCREASE):
        print("..." + str(currentEpochs) + " epochs...")
        # train model
        model.fit(X_train_scaled,trainY,epochs=EPOCH_INCREASE,verbose=0)

        # get results
        bce = BinaryCrossentropy(from_logits=False)
        y_train_hat = model.predict(X_train_scaled)
        y_val_hat = model.predict(X_val_scaled)
        y_train_hat_cat = 1*(y_train_hat > 0.5)
        y_val_hat_cat = 1*(y_val_hat > 0.5)

        # print results
        with open('result.txt', 'a') as fp:
            fp.write("\n\nModel: EPOCHS: "+str(currentEpochs)+" DENSELAYERS: "+str(nodeList))
            fp.write("\nResults: ")
            fp.write("\nTraining Loss: ")
            fp.write(str(bce(trainY.reshape(-1,1), y_train_hat).numpy()))
            fp.write('\n')
            fp.write("Validation Loss: ")
            fp.write(str(bce(valY.reshape(-1,1), y_val_hat).numpy()))
            fp.write("\n--------TRAINING---------\n")
            fp.write(classification_report(trainY,y_train_hat_cat))
            fp.write("\n-------VALIDATION--------\n")
            fp.write(classification_report(valY,y_val_hat_cat))

# store results for different models
numModels = 0
nodeListList = [[1],[1,1],[2,1],[4,1],[16,1],[4,2,1],[8,2,1],[16,4,1],[32,8,1],[256,16,1]]
for nodeList in nodeListList:
    numModels += 1
    tryModel(100,nodeList)
    print(str(numModels)+' models finished!!!')

    

