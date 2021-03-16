#!/usr/bin/env python
# coding: utf-8

# # Techigai DS Assignment - Fake News Detection Classifier

# ## Problem Statement:

# It has become extremely easy to propagate any information to the masses, thus making
# Fake News a potential threat to public discourse. We would like you to build a model that
# when given a news article should determine the relevance of the body and the claim and
# classify the article accordingly into four categories: ‘agree’, ‘disagree’, ‘discuss’ and
# ‘unrelated’. Classes ‘disagree’ and ‘unrelated’ provide strong evidence that the news is fake, ‘agree’
# indicates that the news is genuine

# In[3]:


#Importing all the required Libraries


# In[4]:


import os
import pandas as pd
import numpy as np

#Import Regular Expression
import re

import nltk


# In[5]:


#Read the Data
train_data = pd.read_csv("FakeNewsData.csv")


# In[6]:


train_data.head()


# Data Cleaning -

# In[7]:


#Replacing Column name by Position
train_data.rename(columns={ train_data.columns[0]: "id" }, inplace = True)


# In[8]:


train_data.head()


# In[9]:


train_data.shape


# In[10]:


train_data.dtypes


# In[11]:


train_data.Stance.value_counts()


# ### Feature Engineering

# In[9]:


#Fix levels of categorical variable by relevance of the body and the claim


# In[12]:


# Check levels of Stance. Is there anything wrong?
print(train_data.Stance.value_counts())


# In[13]:


# clean up the relevant level 
train_data.replace(['unrelated'], 'disagree', inplace=True)


# In[14]:


train_data.replace(['discuss'], 'agree', inplace=True)


# In[15]:


train_data.Stance.value_counts()


# In[16]:


train_data.head()


# In[17]:


#Necessary Import
from sklearn.preprocessing import LabelEncoder


# In[18]:


le = LabelEncoder()


# In[19]:


train_data.Stance = le.fit_transform(train_data.Stance)


# In[20]:


train_data['Stance'] = train_data['Stance'].astype('category')


# In[21]:


train_data.head()


# **Stance :** A Label that marks the news article as potentially unreliable.
# 

# 1 : disagree - provide strong evidence that the news is fake 
# 

# 0 : agree - indicates that the news is genuine.

# In[23]:


train_data.Stance.value_counts()


# In[24]:


#Dropping the Headline and Body Word Count Columns
train_data.drop('Headline Word Count',axis = 1, inplace = True)
train_data.drop('Body Word Count', axis = 1,inplace = True)


# Getting Independent Features -

# In[25]:


y_data = train_data['Stance']


# In[26]:


X_data = train_data.drop('Stance', axis = 1)


# In[27]:


X_data.head()


# In[28]:


y_data.head()


# In[29]:


#Total Number of records
train_data.shape


# # Text Preprocessing

# In[30]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[31]:


train_data.isna().sum()


# In[32]:


train_data = train_data.dropna()


# In[33]:


train_data.shape


# In[34]:


articles = train_data.copy()


# In[35]:


articles.head()


# In[36]:


articles['Headline'][3]


# In[37]:


articles.dtypes


# In[40]:


# Custom Code to Cleaning up all the messages,all the texts with special words,Filtering out the Stop words and Stemming.


# In[38]:


#Data Preprocessing and Prep'n
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = [] #Empty corpus
for i in range(0, len(articles)):
    review = re.sub('[^a-zA-Z]', ' ', articles['Headline'][i])
    review = review.lower()
    review = review.split()
# review will be having the list of words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[39]:


corpus


# ### Using Count Vectorizer

# In[41]:


##Applying Count Vectorizer
#Creating the Bag of Words Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features =  5000,ngram_range = (1,3))
X = cv.fit_transform(corpus).toarray()


# In[42]:


## TFidf Vectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
#X=tfidf_v.fit_transform(corpus).toarray()


# In[43]:


X.shape


# In[44]:


y = articles['Stance']


# ### Train Test Split

# In[45]:


# Divide the Data into Train-Test-Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.33,random_state = 0)


# In[46]:


cv.get_feature_names()[:20]


# In[47]:


#Check for the parameters used in CountVectorizer
cv.get_params()


# In[48]:


#Check how the Vectors are formed on your Data
count_df = pd.DataFrame(X_train,columns = cv.get_feature_names())


# In[49]:


count_df.head()


# In[50]:


#To Visualize the Confusion Matrix
import matplotlib.pyplot as plt


# In[51]:


def plot_confusion_matrix(cm , classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    """
    This Function prints and plots Confusion Matrix
    
    """
    plt.imshow(cm, interpolation = 'nearest',cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm  =  cm.astype('float') / cm.sum(axis = 1)[:,np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')

    thresh = cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, cm[i,j]),
        horizontalalignment = "center"
        color = "white" if cm[i,j] > thresh else "black"
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Model Building

# ## Multinomial NB Algorithm

# In[53]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()


# In[54]:


from sklearn import metrics
import numpy as np
import itertools


# In[55]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes = ['disagree','agree'])


# ### Validation Metric

# In[56]:


f1_score = metrics.f1_score(y_test,pred)
print("f1_score:   %0.3f" % f1_score)


# ## Multinomial Classifier with Hyperparameter Tuning

# In[57]:


classifier=MultinomialNB(alpha=0.1)


# In[58]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[59]:


## Get Features names
feature_names = cv.get_feature_names()


# In[60]:


classifier.coef_[0]


# In[61]:


### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# In[62]:


### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]

