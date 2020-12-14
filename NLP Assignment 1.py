#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
path = "Desktop/Food_Inspections.csv"
df = pd.read_csv(path)
df = df[df.Results == 'Fail']
sum(df['Violations'].isnull()) # there are 3139 NaNs in Violation columns


# In[2]:


df.dropna(subset = ['Violations'], inplace=True)
sum(df['Violations'].isnull()) #all the NaNs are dropped


# In[3]:


import re
x = df["Violations"].reset_index(drop=True).str.split("|")  #split each violations
from itertools import chain
x = list(chain.from_iterable(x.tolist()))  #unlist the nested list


# In[4]:


# since some of the violation only have a regulation code and regulation description but no comments, to better detect all the description, we manually add the string "- Comments:" to descriptions that does not have a comment.
y=[]
for i in range(len(x)):
    if re.search("\- Comments:", x[i]): 
        y.append(x[i])
    else:
        y.append(x[i]+"- Comments:")


# In[5]:


pattern = "\d+.(.*?)\- Comments:"
description = [re.search(pattern, y[i]).group(1) for i in range(len(y))]


# In[6]:


description = [description[i].strip() for i in range(len(description))] #remove white space
from collections import Counter
Counter(description).most_common(10)

