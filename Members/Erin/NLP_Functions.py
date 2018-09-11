import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string, re, random
from random import sample
import itertools

pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',80)

#Spacy libraries:
import spacy
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.corpus import stopwords

#GenSim Doc2Vec libraries:
import gensim
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import os
cores = multiprocessing.cpu_count()


#-----Load CSV file(s) with combined text (one row per restaurant)---#

df_tips = pd.read_csv("./df_tips.csv")
#df_reviews = pd.read_csv("./df_reviews.csv")

Rest_A = user_restaurant(df_tips)
Rest_B = user_restaurant(df_tips)

###-----DEFINE FUNCTIONS------###
###---------------------------###

#-----Choose users' favorite restaurants:---#
def user_restaurant(df):
  choice = df.sample() #choose random sample from df
  #choice = df[(df.business_id == 'xM37qm9Wbc-hOAS7-Xse7g')]  #input specific restaurant business_id
  choice = choice.filter(["name","business_id","text"])
  print("User's favorite restaurant is:",choice['name'].values[0],"\n")
  return(choice)

#---------Spacy NLP-----------#
#ensure df has column of text (reviews or tips), business_id, name
#df_tips['text'] = df_tips['tip']

def spacy_nlp(df, Rest_A, Rest_B):
  df = df.filter(["name","business_id","text"])
  df['nlp_text'] = df['text'].apply(lambda x: nlp(x))   # <---long processing time
  AB_tipcombo = Rest_A['text'].values[0] + Rest_B['text'].values[0]
  AB_restaurants = [Rest_A['name'].values[0], Rest_B['name'].values[0]]
  #Apply NLP tokenization to combined tip text for chosen restaurants A & B
  AB_tiptoken = nlp(AB_tipcombo)
  df = df.drop(df[(df.business_id == Rest_A.business_id.values[0]) | \
                  (df.business_id == Rest_B.business_id.values[0])].index)
  df['similarity'] = df['nlp_text'].apply(lambda x: x.similarity(AB_tiptoken))
  #Display top 5 recommended restaurants
  df.sort_values('similarity',ascending=False).head(20)





#---------Doc2Vec NLP---------#
df_tips = df_tips.filter(["name","business_id","categories","tip","text"])
df_tips = df_tips.drop(columns=(['text']))
