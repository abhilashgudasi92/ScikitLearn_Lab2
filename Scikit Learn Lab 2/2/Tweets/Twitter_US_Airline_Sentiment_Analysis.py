
# coding: utf-8

# In[1]:


# 2. Working with Text Data [20 pts]
# Text data is the most common form of data and is widely used in machine learning. In this section,
# you will learn techniques for pre-processing and model building using text data.
# First of all, you will need to work through some examples and become familiar with text processing techniques.
# Below is the link to a tutorial on SciKit Learn:
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


# In[2]:


################################################################################################################################
#
# Dataset: Twitter US Airline Sentiment
# Reference: https://www.kaggle.com/crowdflower/twitter-airline-sentiment
# Description: Contains whether the sentiment of the tweets in this set was positive, neutral, or negative for six US airlines
#
################################################################################################################################


# In[8]:


########################################################################
#	Loading and inspecting the dataset
########################################################################

# Reading dataset
import pandas as pd
categories = ['positive', 'negative','neutral']
import os
dirName = os.path.dirname(os.path.abspath("Tweets.csv"))
Tweet = pd.read_csv(dirName + "\Tweets.csv")

#Tweet = pd.read_csv("C:/Users/Arun/Desktop/Abhi/UTD/Spring'18/Machine learning/Lab/2/Tweets/Tweets.csv")


# In[9]:


Tweet.shape


# In[10]:


# Hence, we have 14640 samples and 15 features


# In[11]:


Tweet.head()


# In[12]:


Tweet = Tweet.drop(['airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline_sentiment_gold', 'negativereason_gold', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone'], axis = 1)


# In[13]:


Tweet.head()


# In[14]:


Tweet.shape


# In[15]:


# Hence, we have dropped few of the columns and we are left with 6 features only.


# In[16]:


# Groupby airline, and reference the airline_sentiment column and then extract total count
print(Tweet.groupby('airline')['airline_sentiment'].count())


# In[17]:


# groupby both airlines and airline_sentiment and extract total count
print(Tweet.groupby(['airline','airline_sentiment']).count().iloc[:,0])


# In[18]:


# From above results, we observe that United airways has more negative sentiments. But, we have assumed that correctly identified 
# airline being referenced in the tweet text and hence we need to clean the data


# In[19]:


#Here we checking airline mismatch tweets
observation = list(Tweet.iloc[6750:6755,2])
#print(observation)
tweet_text = list(Tweet.iloc[6750:6755,5])
#print(tweet_text)
print("Misclassified data:")
for pos, item in enumerate(observation):
    print('Airline as compiled: ' + str(item))
    print('The actual tweet text: ')
    print(tweet_text[pos], '\n''\n')


# In[20]:


# Trying to replace mismatch airline data, hence we will fetch first tag from the tweet text 
observation = list(Tweet.iloc[:,2])
tweet_text = list(Tweet.iloc[:,5])
import re
for pos, item in enumerate(observation):
    a =re.findall('\@[A-Za-z]+',tweet_text[pos],flags=0)[0]
    if(a.lower() != "@" + str(item).replace(" ", "").lower() and a.lower() != "@" + str(item).replace(" ", "").lower()+"air"): 
        Tweet.iloc[pos,2] = Tweet.iloc[pos,2].replace(Tweet.iloc[pos,2],a[1:])


# In[21]:


#After preprocessing the misclassified data
observation = list(Tweet.iloc[6750:6755,2])
tweet_text = list(Tweet.iloc[6750:6755,5])
print("Misclassified data fixed:")
for pos, item in enumerate(observation):
    print('Airline as compiled: ' + str(item))
    print('The actual tweet text: ')
    print(tweet_text[pos], '\n''\n')


# In[23]:


print(Tweet.groupby('airline')['airline_sentiment'].count())


# In[24]:


observation = list(Tweet.iloc[:,2])
for pos, item in enumerate(observation):
    if(str(item).lower() == 'jetblue'):
        Tweet.iloc[pos,2] = 'JetBlue'
#print(Tweet.groupby('airline')['airline_sentiment'].count())


# In[25]:


#Removing unwnated usertags in airline column 
airline_list = ['virginamerica','united','southwest','american','jetblue','usairways']
observation = list(Tweet.iloc[:,2])
posList = []
for pos, item in enumerate(observation):
    if(str(item).replace(" ", "").lower() not in airline_list):
        posList.append(pos)
Tweet.drop(Tweet.index[posList],inplace=True)


# In[26]:

print("Final Cleaned data:")
print(Tweet.groupby('airline')['airline_sentiment'].count())


# In[27]:


########################################################################
# 	Visualization of data
########################################################################

import matplotlib.pyplot as plt 
Index = [1, 2, 3]
plt.bar(Index,Tweet.airline_sentiment.value_counts())
plt.xticks(Index,['negative','neutral','positive'],rotation=45)
plt.ylabel('Numbber of Tweets')
plt.xlabel('Sentiment expressed in tweets')


# In[28]:


print(Tweet.groupby(["airline","airline_sentiment"]).size().unstack())


# In[29]:


########################################################################
#	Splitting data into train/test
########################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Tweet.text, Tweet.airline_sentiment, test_size=0.25, random_state=0)


# In[30]:


#######################################################
# Extracting features from text files
#######################################################

# In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors.

# Tokenizing text with scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)


# In[31]:


count_vect.get_feature_names()


# In[32]:


# Calculating tf (term frequencies => occurence of a word / total number of words in the document)
# Another refinement on top of tf is to downscale weights for words that occur in many documents in the corpus and
# are therefore less informative than those that occur only in a smaller portion of the corpus.
# This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.

# Both tf and tf–idf can be computed as follows:

#from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)


# In[34]:


# Alternative to above code snippet
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[35]:


#################################################################################################################################
# Training a classifier -> we can train a classifier to try to predict the sentiment of a tweet
#################################################################################################################################
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[36]:


#################################################################################################################################
# Prediction
#################################################################################################################################

X_new_counts = count_vect.transform(x_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)


# In[37]:


#######################################################
# Building a pipeline ->In order to make the vectorizer => transformer => classifier easier to work with, 
# scikit-learn provides a Pipeline class that behaves like a compound classifier:
#######################################################
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])


# In[38]:


text_clf.fit(x_train, y_train)


# In[39]:


#######################################################
# Evaluation of the performance on the test set -> Evaluating the predictive accuracy of the model
#######################################################
import numpy as np
predicted = text_clf.predict(x_test)
np.mean(predicted == y_test)


# In[40]:


# confusion matrix
from sklearn import metrics 
metrics.confusion_matrix(y_test, predicted)


# In[41]:


#Trying out with GridSearch with different parameters
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-1, 1e-2,2e-2)
}


# In[42]:


gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)


# In[43]:


gs_clf = gs_clf.fit(x_train, y_train)


# In[44]:


# best mean score
gs_clf.best_score_ 
print("Accuracy: ", gs_clf.best_score_ )


# In[45]:


predicted = gs_clf.predict(x_test)


# In[46]:


#Confusion matrix after applying GridsearchCV 
metrics.confusion_matrix(y_test, predicted)


# In[47]:


# parameter setting corresponding to above score
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

