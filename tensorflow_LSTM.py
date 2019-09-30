# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:27:24 2019

@author: fkiaie
"""
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 50000
embedding_dim = 16
max_length = 50000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_portion = .8

df = pd.read_csv('data.csv')

#df = df.sample(n=500)
#########################cleaning Data set

print(df.duplicated().sum())

print(df.isnull().sum())
df=df.dropna()
#########################feature engineering 
df['news'] = df.Headline + df.Body
df.Label = df.Label.map({1: 'yes', 0: 'no'})


Max_len=0
temp_len=0

for j in range(len(df)):
	temp_len = len(df.news.iloc[j])
	if Max_len < temp_len:
		Max_len=temp_len

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
	
y=df.Label
X=corpus

##################################splitting data
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)




tokenizer=Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
train_sequences=tokenizer.texts_to_sequences(X_train)
train_padded=pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y)
#
training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))
#model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#    tf.keras.layers.Dense(24, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])
	
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)





def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'acc')
plot_graphs(history, 'loss')


###############################word cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = X[0:20]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'white',
    stopwords = stop_words).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()






#######




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
