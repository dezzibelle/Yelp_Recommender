
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',80)
import string
import re
from random import sample
import itertools


# In[2]:


import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.corpus import stopwords


# ***Skip to LOAD CSV FILE below if pre-processing is complete **

# In[4]:


#Load full dataframe with tips & pre-process to combine all tips for each restaurant & run NLP/tokenization on each
df = pd.read_pickle("./data/df_LVtips.pkl")

df.head()
# In[5]:


df = df.drop(columns = ["bus_id_size"])
df.columns = df.columns.str.replace('attributes_','').str.replace('Restaurants','')


# In[6]:

tips1 = df.groupby('business_id')['tip'].apply(lambda tip: ''.join(tip.to_string(index=False))).str.replace('(\\n)', '').reset_index()


# In[7]:


#Combine text for all tips into one cell for each restaurant
df = df.drop(columns = ['tip'])
df = df.drop_duplicates()

df = pd.merge(df, tips1, on='business_id')


# In[8]:


#remove extra white space in combined tip text
df['tip'] = df['tip'].str.replace('\s+',' ').str.replace('\$', '\\$')


# In[20]:


#Remove HTML formatting
# from IPython.core.display import HTML
# HTML(df.to_html(escape=False))


# In[11]:


#Remove punctuation & apply nlp tokenization to full dataframe of tips

df['tip'] = df['tip'].apply(lambda x: re.sub('[^\w\s]','', x))
df['text'] = df['tip'].apply(lambda x: nlp(x))

df.head()
# ***LOAD (or save) CSV FILE WITH NLP TOKEN COLUMN***

# In[18]:


# SAVE df in csv:  -->Note: re-save with all/relevant columns to run additional filters
df.to_csv("./df_tips_spacy_nlp.csv")
df_tips=df.filter(["name","business_id","categories","tip","text"])

# LOAD csv file:
df_tips = pd.read_csv("./df_tips_spacy_nlp.csv")
df_tips=df_tips.filter(["name","business_id","categories","tip","text"])

# In[42]:
df_tips.head()

#App User Inputs:  (currently random restaurants sampled from dataset)

Rest_A = df_tips.sample()
Rest_B = df_tips.sample()
print("User A's favorite retaurant is:",Rest_A['name'].values[0],"\n")
print("User B's favorite retaurant is:",Rest_B['name'].values[0])

AB_tipcombo = Rest_A['tip'].values[0] + Rest_B['tip'].values[0]
AB_restaurants = [Rest_A['name'].values[0], Rest_B['name'].values[0]]

#Apply NLP tokenization to combined tip text for chosen restaurants A & B
AB_tiptoken = nlp(AB_tipcombo)


# **Run model with the 2 user-input restaurants & compare with all others in the database**
#
# Note: this is where a filter could be applied to df_tips before running NLP comparison
# (i.e. distance, cuisine, eat-in/takeout/delivery/drive-thru)

# In[43]:


df = df_tips.drop(df_tips[(df_tips.business_id == Rest_A.business_id.values[0]) | (df_tips.business_id == Rest_B.business_id.values[0])].index)

df['similarity'] = df['text'].apply(lambda x: x.similarity(AB_tiptoken))

#Display top 5 recommended restaurants
df.filter(["name","business_id","categories","similarity"]).sort_values('similarity',ascending=False).head(5)


# In[44]:


Rest_A.filter(["name","business_id","categories"])


# In[45]:


Rest_B.filter(["name","business_id","categories"])
