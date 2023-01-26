#!/usr/bin/env python
# coding: utf-8

# # Model To Predict The Sentiment of a Tweet
# The aim of this project is to create a model which should be able to detect the sentiment (positive or negative) of the inputted tweet.

# In[40]:


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
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import BinaryCrossentropy


# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[42]:


# reading the data
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
df = df.drop([1,2,3,4], axis=1)
df.columns = ['sentiment', 'text']
# df now consists of only the text and associated sentiment
print(df.head())
print(df.shape)


# In[43]:


# 0 = negative, 1 = positive
df.sentiment = df.sentiment.apply(lambda x: 1 if (x == 4) else 0)
df.tail()


# In[44]:


val_count = df.sentiment.value_counts()
plt.figure(figsize=(4,4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")


# In[7]:


# Split data into training set (includes validation set) and testing set.
valSize = 0.15
testSize = 0.1

trainValX, testX, trainValY, testY = train_test_split(df.text, df.sentiment, test_size=testSize, random_state=4)


# In[8]:


# text-preprocessing: removing hyperlinks, hashtags and twitter mentions
def cleanText(text):
    text = re.sub(r'https?://[^\s\n\r]+','',text.lower())
    text = re.sub(r'@[^\s\n\r]+','',text)
    text = re.sub(r'#','',text)
    # remove any unnecessary double spaces
    text = re.sub(r'  ',' ',text).strip()
    return text
trainValX = trainValX.apply(cleanText)


# In[ ]:


print(trainX.head())
print("------")
print(trainY.head())
print(trainX.iat[0])


# In[9]:


# tokenizing tweets
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
trainValX = trainValX.apply(tokenizer.tokenize)


# In[10]:


# removing stop words and punctuations (cleaning the data up a bit more)
stopwords_english = stopwords.words('english')

def cleanStopWords(token):
    cleaned_token = []
    for word in token:
        if(word not in stopwords_english and word not in string.punctuation):
            cleaned_token.append(word)
    return cleaned_token

trainValX = trainValX.apply(cleanStopWords)


# In[11]:


# stemming the words in the tweet tokens
stemmer = PorterStemmer()

def cleanStemWords(token):
    cleaned_token = []
    for word in token:
        cleaned_token.append(stemmer.stem(word))
    return cleaned_token

trainValX = trainValX.apply(cleanStemWords)


# In[12]:


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

frequencies = build_freqs(trainValX, trainValY)


# In[13]:


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
features = calculate_features(trainValX,frequencies)
features


# In[14]:


# since the data is processed, separate the data into a training and validation set
trainX, valX, trainY, valY = train_test_split(features, trainValY.to_numpy(), test_size=valSize, random_state=4)


# In[15]:


# visualizing our features
sns.scatterplot(features[:,0],features[:,1],hue=trainValY)
plt.xlabel('negative frequencies of the tweet',fontsize=20)
plt.ylabel('positive frequencies of the tweet',fontsize=20)
plt.xlim(0,5000)
plt.ylim(0,5000)


# In[16]:


# scale the data
scaler = preprocessing.StandardScaler()
scaler.fit(trainX)
X_train_scaled = scaler.transform(trainX)
X_val_scaled = scaler.transform(valX)


# In[17]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[19]:


# creating the model -- logistic regression
model = 0 
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy')
model.fit(X_train_scaled,trainY,epochs=40,verbose=0)
J_list = model.history.history['loss']
plt.plot(J_list)
model = 0 
model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy')
model.fit(X_train_scaled,trainY,epochs=40,verbose=3)
J_list = model.history.history['loss']
plt.plot(J_list)


# In[21]:


bce = BinaryCrossentropy(from_logits=False)
y_train_hat = model.predict(X_train_scaled)
print("Training:")
print(bce(trainY.reshape(-1,1), y_train_hat).numpy())
print('---')
print("Validation:")
y_val_hat = model.predict(X_val_scaled)
print(bce(valY.reshape(-1,1), y_val_hat).numpy())


# In[23]:


y_train_hat_cat = 1*(model.predict(X_train_scaled) > 0.5)
print(classification_report(trainY,y_train_hat_cat))


# In[24]:


y_val_hat_cat = 1*(model.predict(X_val_scaled) > 0.5)
print(classification_report(valY,y_val_hat_cat))


# In[32]:


# reading the data
trial = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
trial = trial.drop([1,2,3,4], axis=1)
trial.columns = ['sentiment', 'text']
#df now consists of only the text and associated sentiment


# In[33]:


print(trial.text.head())


# In[39]:


import random
random_idx_list = [random.randint(1,len(df.text)) for i in range(100)] 
print(df.loc[random_idx_list,:].head(50))


# In[ ]:




