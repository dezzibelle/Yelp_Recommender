
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
import random
from random import sample
import itertools

import gensim
from gensim.models import Word2Vec
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import multiprocessing
import os
cores = multiprocessing.cpu_count()


# In[3]:


# LOAD csv file:  (also has tokenized column of tip text from spacy nlp)
df_tips = pd.read_csv("./df_tips_spacy_nlp.csv")
df_tips = df_tips.filter(["name","business_id","categories","tip","text"])
df_tips = df_tips.drop(columns=(['text']))
# In[5]:


#App User Inputs:  (currently random restaurants sampled from dataset)

Rest_A = df_tips[(df_tips.business_id == 'lfXfxBms5z1nwzkxxLFBWg')]
Rest_B = df_tips[(df_tips.business_id == 'DbEszO3wk1xVmN3pCPob2g')]    #= df_tips.sample()
print("User A's favorite retaurant is:",Rest_A['name'].values[0],"\n")
print("User B's favorite retaurant is:",Rest_B['name'].values[0])


# In[26]:


#Create combination df with combined restaurant text & remove individual rows for restaurants A & B

AB_tipcombo = Rest_A['tip'].values[0] + Rest_B['tip'].values[0]
AB_restaurants = [Rest_A['name'].values[0], Rest_B['name'].values[0]]

df_combo = df_tips.filter(["name","business_id","tip"],axis=1)
df_combo = df_combo.append({'name': AB_restaurants, 'business_id': '', 'tip': AB_tipcombo}, ignore_index=True)
df_combo = df_combo.drop(df_combo[(df_combo.business_id == Rest_A.business_id.values[0]) | (df_combo.business_id == Rest_B.business_id.values[0])].index)


# In[28]:


# MyDocs reading from df_combo data frame
class MyDocs(object):
    def __iter__(self):
        for i in range(df_combo.shape[0]):
            yield TaggedDocument(words=simple_preprocess(df_combo.iloc[i,-1]), tags=['%s' % df_combo.iloc[i,0]])


# In[29]:


get_ipython().run_cell_magic('time', '', '\nif not os.path.exists(\'models/doc2vec.model\'):\n    print("start training doc2vec model...")\n    documents = MyDocs()\n    doc2vec_model = Doc2Vec(dm=1, dbow_words=1, vector_size=200, window=8, min_count=20, workers=cores)\n    doc2vec_model.build_vocab(documents)\n    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)\n    if not os.path.exists(\'models\'):\n        os.makedirs(\'models\')\n        doc2vec_model.save(\'models/doc2vec.model\')\n    else:\n        doc2vec_model.save(\'models/doc2vec.model\')\nelse:\n    doc2vec_model = Doc2Vec.load(\'models/doc2vec.model\')')


# In[30]:


doc2vec_model.docvecs.most_similar(AB_restaurants, topn=20)
