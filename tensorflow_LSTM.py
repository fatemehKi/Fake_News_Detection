# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:27:24 2019

@author: fkiaie
"""
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('data.csv')

#########################cleaning Data set
df=df.dropna()

#########################feature engineering 
df['news'] = df.Headline + df.Body

#########################
stop_words = set(stopwords.words('english'))
#lemmatize and stem
ps = PorterStemmer()
lem = WordNetLemmatizer()
#dataset=pd.read_table("sms_spam_ham.tsv", header=None, names=['label_y', 'sms'])

corpus=[]

i=0
for i in range(len(df)):
    text = df.news.iloc[i]
    text = text.lower() #changes evrything lower case
    nopunct_text = re.sub('[^a-z0-9]',' ',text) #remove non alphanumeric characters
    tokens = WhitespaceTokenizer().tokenize(nopunct_text)
    filtered = [ps.stem(lem.lemmatize(w)) for w in tokens if w not in stop_words]
    filtered_text=' '.join(filtered)
    corpus.append(filtered_text)
	
y=df['Label'].values 
X=corpus

##################################splitting data
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
tokenizer=Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
sequences=tokenizer.texts_to_sequences(X_train)
padded=pad_sequences(sequences, padding='post')





###################################################################
################ loading dataset
def load_file(file):
    '''loading csv fel to pd dataframe'''
    df=pd.read_csv(file)
    return df



################ Data Cleaning
def cleaning(df):
    '''handling missing values using backward filling'''
    print(df.isnull().sum())    
    if (df.isnull().sum().any() !=0):
        df=df.dropna(how='all')
        df=df.fillna(method='bfill', inplace=True)
    return df

################ Dropping unrelated features
def rmv_usls(df, col):
    '''droping features that are not related'''
    df=df.drop(col, axis=1)
    return df
