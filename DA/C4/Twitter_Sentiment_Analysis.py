#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[99]:


get_ipython().system('pip install gensim')
get_ipython().system('pip install worldcloud')


# In[100]:


get_ipython().system('pip install nltk')


# In[101]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# In[104]:


train = pd.read_csv('train_tweet.csv')
test = pd.read_csv('test_tweets.csv')


# In[105]:


print(train.shape)
print(test.shape)


# In[106]:


train.head()


# In[107]:


test.head()


# In[108]:


train.isnull().any()
test.isnull().any()


# #### Checking out the negative comments from train dataset

# In[109]:


train[train['label']==0].head(5)


# #### Checking out the positive comments from tarin dataset

# In[110]:


train[train['label']==1].head(5)


# In[111]:


train['label'].value_counts().plot.bar(color = 'green', figsize = (6, 4))


# ### Observations:
#     1. Negative tweets exceed positive tweets largely in number

# #### Checkng the distribution of tweets in the data

# In[112]:


length_train = train['tweet'].str.len().plot.hist(color = 'pink', figsize = (6, 4))
length_test = test['tweet'].str.len().plot.hist(color = 'purple', figsize = (6, 4))


# #### Adding column to represent length of the tweet

# In[113]:


train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()


# In[114]:


train.head(10)


# ##### Numeric data analysis

# In[115]:


train.groupby('label').describe()


# ##### Observing the length of tweets generally used

# In[116]:


train.groupby('len').mean()['label'].plot.hist(color = 'magenta', figsize = (6, 4),)
plt.title('variation of length')
plt.xlabel('Length')
plt.show()


# #### CountVectorization
# ##### returns each unique word as a feature with the count of number of times that word occurs.

# In[117]:


from sklearn.feature_extraction.text import CountVectorizer


# In[118]:


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train.tweet)


# In[119]:


sum_words = words.sum(axis=0)


# In[120]:


words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)


# In[121]:


frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])


# In[122]:


frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")


# #### WordCloud

# In[123]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))


# In[124]:


plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)


# In[125]:


normal_words =' '.join([text for text in train['tweet'][train['label'] == 0]])


# In[126]:


wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()


# In[127]:


negative_words =' '.join([text for text in train['tweet'][train['label'] == 1]])


# In[128]:


wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()


# #### Collecting hashtags

# In[129]:


import re
def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# #### extracting hashtags from non racist/sexist tweets

# In[130]:


HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])


# ##### extracting hashtags from sexist/racist tweets

# In[131]:


HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])


# ##### unnesting list

# In[132]:


HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[133]:


import nltk
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})


# ##### selecting top 20 most frequent hashtags

# In[134]:


d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[135]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})


# In[136]:


d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# ##### tokenizing te words present in training set

# In[137]:


tokenized_tweet = train['tweet'].apply(lambda x: x.split()) 


# ##### importing gensim

# In[138]:


import gensim


# ##### creating a word to vector model

# In[139]:


model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)


# In[140]:


model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']), epochs=20)


# In[141]:


model_w2v.wv.most_similar(positive = "dinner")


# In[142]:


model_w2v.wv.most_similar(positive = "cancer")


# In[143]:


model_w2v.wv.most_similar(positive = "apple")


# In[144]:


model_w2v.wv.most_similar(negative = "hate")


# In[145]:


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence


# In[146]:


def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output


# ##### label all tweets

# In[147]:


labeled_tweets = add_label(tokenized_tweet)


# In[148]:


labeled_tweets[:6]


# ##### removing unwanted patterns from data

# In[149]:


import re
import nltk


# In[150]:


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[151]:


train_corpus = []


# In[152]:


for i in range(0, 31962):
  review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
  review = review.lower()
  review = review.split()


# In[153]:


ps = PorterStemmer()


# ##### Stemming

# In[154]:


review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


# ##### Joining them back with space

# In[155]:


review = ' '.join(review)
train_corpus.append(review)


# In[156]:


test_corpus = []


# In[157]:


for i in range(0, 17197):
  review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
  review = review.lower()
  review = review.split()


# In[158]:


ps = PorterStemmer()


# ##### stemming

# In[159]:


review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


# ##### Joining them back with spaces

# In[160]:


review = ' '.join(review)
test_corpus.append(review)


# ##### Creating bag of words

# In[170]:


from sklearn.feature_extraction.text import CountVectorizer


# In[171]:


cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]


# In[172]:


print(x.shape)
print(y.shape)


# In[164]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x_test = cv.fit_transform(test_corpus).toarray()

print(x_test.shape)


# ##### Splitting training data into train and valid sets

# In[168]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:



