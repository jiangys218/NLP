#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. from Assignment 1
import pandas as pd
path = "Desktop/Food_Inspections.csv"
df = pd.read_csv(path)
df = df[df.Results == 'Fail']
df.dropna(subset = ['Violations'], inplace=True)

import re
x = df["Violations"].reset_index(drop=True).str.split("|")  #split each violations
from itertools import chain
x = list(chain.from_iterable(x.tolist()))  #unlist the nested list

# since some of the violation only have a regulation code and regulation description but no comments, to better detect all the description, we manually add the string "- Comments:" to descriptions that does not have a comment.
y=[]
for i in range(len(x)):
    if re.search("\- Comments:", x[i]): 
        y.append(x[i])
    else:
        y.append(x[i]+"- Comments:")
        
pattern = "\d+.(.*?)\- Comments:"
description = [re.search(pattern, y[i]).group(1) for i in range(len(y))]
description = [description[i].strip() for i in range(len(description))] #remove white space


# In[10]:


des= " ".join(description)


# In[12]:


#2. Tokenize each regulation description
#3. Find top-10 tokens (for the whole table)
import nltk as nltk
import nltk.corpus  
from nltk.text import Text

des = nltk.tokenize.word_tokenize(des)
fdist3 = nltk.FreqDist(des)
fdist3.most_common(10)


# In[19]:


#4. Clean data: convert to lower case, remove stopwords, punctuation, numbers, etc
#5. Find top-10 tokens again

stopwords = set(nltk.corpus.stopwords.words('english'))

# Remove single-character tokens (mostly punctuation)
words = [word for word in des if len(word) > 1]

# Remove numbers
words = [word for word in words if not word.isnumeric()]

# Remove punctuation
words = [word for word in words if word.isalpha()]

# Lowercase all words (default_stopwords are lowercase too)
words_lc = [word.lower() for word in words]

# Remove stopwords
words_lc = [word for word in words_lc if word not in stopwords]

fdist5 = nltk.FreqDist(words_lc)
fdist5.most_common(10)


# In[21]:


#6. Find top-10 tokens after applying Porter stemming to the tokens obtained in step 4.

porter = nltk.PorterStemmer()
porterstem = [porter.stem(t) for t in words_lc]
fdist6 = nltk.FreqDist(porterstem)
fdist6.most_common(10)


# In[22]:


#7. Find top-10 tokens after applying Lancaster stemming to the tokens obtained in step 4.

lancaster = nltk.LancasterStemmer()
lancasterstem = [lancaster.stem(t) for t in words_lc]
fdist7 = nltk.FreqDist(lancasterstem)
fdist7.most_common(10)


# In[23]:


#8. Find top-10 tokens after applying lemmatization to the tokens obtained in step 4.

wnl = nltk.WordNetLemmatizer()
lemma = [wnl.lemmatize(t) for t in words_lc]
fdist8 = nltk.FreqDist(lemma)
fdist8.most_common(10)


# In[ ]:


#Compare top-10 tokens obtained in 3, 5, 6, 7, 8.
#We started with tokenize each regulation description without any cleaning and pre-processing (#3), and as expected, there are stopwords and punctuation in the top 10 tokens list, which is not a good solution. After realizing this problem, we perform some text cleaning including removing stopwords, numbers, punctuation, etc. (#5) and the top 10 tokens result has improved significantly. We can see that the top 10 tokens are mostly related cleaness (maintained, clean, cleanring, surface, good), food, and equipment (installed, equipment, properly, good, constructed). Next, we perform porter stemming on the clean token (#6). However, since stemming method tends to recognize and summary tokens based on their roots, we can see some of our top 10 tokens are mis-spelled after stemming ("properli", "instal","surfac"), which might cause confusion. Even though the result from porter stemming is very similar to the result in part 5, one of benefits from using stemming method is it generalize "clean" and "cleaning" into one token, and introduce an new top 10 token that is not in the  part 5, which is "method". Next, in Lancaster stemming (#7), we can see a very similar result to porter stemming, except that Lancaster stemming generalize the tokens further, which leads to confusion in some of the top 10 tokens ("cle", "prop"). Lastly, we tried lemmatization (#8). Lemmatization provides the best top 10 tokens result because it considers tokens roots ("clean" and "cleaning" considered into one token) but did not cause confusion in the generalized token/ no mis-spelling.     

