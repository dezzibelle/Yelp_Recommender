#%%
import json
import pandas as pd
import numpy as np
import itertools
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
import seaborn as sns
import string, re, random
from random import sample
from math import log
import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import spacy
import timeit
from sklearn.cluster import KMeans
sns.set()
from numpy.random import randint
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from sklearn import manifold
#GenSim Doc2Vec libraries:
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import multiprocessing as mp
import os
import pickle
cores = multiprocessing.cpu_count()
#%%



#------Display up to 200 columns & all content within each column

#%%
pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',50)
#%%

#------Load dataframe of restaurant reviews for restaurants with >25 reviews

df = pd.read_pickle("./df_LVrestaurants25samples.pkl")

#sample 25 reviews from each restaurant
#df = df1.groupby('business_id').apply(lambda x: x.sample(25))

#-------Create smaller dataframe (fewer columns) for simplified viewing
dfR = df.filter(["name","address","business_id","review_id","text"])
dfR = dfR.set_index('review_id')
dfR = dfR.reset_index()

#-------Creating a businesses dataframe
column_names=['name', 'address']
sample_name=dfR.groupby('business_id')[column_names].agg({"name": lambda x: x.unique(), "address":lambda x: x.unique()})
sample_name_df=pd.DataFrame(sample_name)

businesses=sample_name_df
#-------------------------------------------------------------------------------
#-----Set up & train Doc2Vec model on 25 reviews for each restaurant in dataframe
#-------------------------------------------------------------------------------
class MyReviews(object):
    def __iter__(self):
        for i in range(dfR.shape[0]):
            yield TaggedDocument(words=simple_preprocess(dfR.iloc[i,-1]), tags=[str.format(dfR.iloc[i,0])])

assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

if not os.path.exists('models/doc2vec.model'):
    print("Start training doc2vec model...")
    documents = MyReviews()
    doc2vec_model = Doc2Vec(dm=1, dbow_words=1, vector_size=200, window=8, min_count=5, workers=cores)
    doc2vec_model.build_vocab(documents)
    print("Vocab built...")
    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    print("Minutes training time:",doc2vec_model.total_train_time/60)
    print("Model trained!")
    if not os.path.exists('models'):
        os.makedirs('models')
        doc2vec_model.save('models/doc2vec.model')
    else:
        doc2vec_model.save('models/doc2vec.model')
else:
    doc2vec_model = Doc2Vec.load('models/doc2vec.model')
#----end of model training

#-------------------------------------------------------------------------------
#-----Test dataframe with 2 sampled restaurants------

Rest_A = df.sample(1).filter(["name","address","business_id"]).set_index('business_id').reset_index()
print(Rest_A.name.values[0],",",Rest_A.address.values[0])
Rest_B = df.sample(1).filter(["name","address","business_id"]).set_index('business_id').reset_index()
print(Rest_B.name.values[0],",",Rest_B.address.values[0])

RA_df = dfR[dfR.business_id == str.format(Rest_A.business_id[0])]  #dataframe with all Restaurant A's reviews
RB_df = dfR[dfR.business_id == str.format(Rest_B.business_id[0])]  #dataframe with all Restaurant B's reviews

#----Find similarity to each restaurant----------------
RA_sim = pd.DataFrame(doc2vec_model.docvecs.most_similar(RA_df.iloc[0:25,0], topn=len(dfR)))
RA_sim.columns = ['review_id','A_sim']   #A_sim is averaged(?) over all 25 reviews for Rest_A
RA_sim = pd.merge(RA_sim,dfR,on="review_id",how="left")
RA_sim = RA_sim.groupby(by=["business_id","name","address"]).agg(['mean','median','min','max']).sort_values(by=("A_sim","median"),ascending=False).reset_index()
RA_sim.head()


RB_sim = pd.DataFrame(doc2vec_model.docvecs.most_similar(RB_df.iloc[0:25,0], topn=len(dfR)))
RB_sim.columns = ['review_id','B_sim']
RB_sim = pd.merge(RB_sim,dfR,on="review_id",how="left")
RB_sim = RB_sim.groupby(by=["business_id","name","address"]).agg(['mean','median','min','max']).sort_values(by=("B_sim","median"),ascending=False).reset_index()
RB_sim.head()

#-------------------------------------------------------------------------------
#-------------- Arrange & sort results

results = pd.merge(RA_sim,RB_sim,on=('business_id','name','address'))
results['AB_sim'] = results[[('A_sim','median'),('B_sim','median')]].mean(axis=1)
results = results.sort_values(by="AB_sim",ascending=False)
#------------------------------------------------------------------------------------------
### Dataframe with statistics for most similar restaurants to BOTH restaurant A & restaurant B
results.head()


#-------------------------------------------------------------------------------
#---------------Find Vectors for all Reviews----------------

review_idlist = dfR["review_id"].tolist()
vectors = []
for i in range(0,len(review_idlist)):
    vectors.append(doc2vec_model.docvecs[i])

Review_vectors = pd.DataFrame({'review_id':review_idlist, 'vectors':vectors})

Review_vectors.to_pickle("ReviewVectors2.pkl")

Review_vectors.head()

#-------------------------------------------------------------------------------
#-----Cosine similarity based on TFIDF
#-------------------------------------------------------------------------------
nlp  = spacy.load('en_core_web_md')

#Lemmatise
def keep_token(t):
    return (t.is_alpha and
            not (t.is_space or t.is_punct or
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = [lemmatize_doc(nlp(doc)) for doc in dfR.text]

#Create a dictionary and filter stop and infrequent words
docs_dict = Dictionary(docs)
docs_dict.filter_extremes(no_below=20, no_above=0.2)
docs_dict.compactify()

#Bag of words for each documents, build TFIDF for each model and compute TF-IDF vector for each document
docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
docs_tfidf  = model_tfidf[docs_corpus]
reviews_tfidf  = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
reviews_tfidf.shape


#Cosine similarity
def find_similar(tfidf_matrix, review_index):
    cosine_similarities = cosine_similarity(tfidf_matrix[review_index:review_index+1], tfidf_matrix).flatten()
    return cosine_similarities

#  Restaurants of choice
Rest_A
Rest_B

rest1_review_ids = dfR[dfR.business_id==str.format(Rest_A.business_id[0])].review_id.tolist()
restaurant_A_review_indicies = dfR.groupby("business_id").indices[str.format(Rest_A.business_id[0])]
restaurant_B_review_indicies = dfR.groupby("business_id").indices[str.format(Rest_B.business_id[0])]


index = pd.MultiIndex.from_tuples(zip(dfR.business_id, dfR.review_id), names=['business_id', 'review_id'])
cosine_df = pd.DataFrame(index=index)


for review_index in restaurant_A_review_indicies + restaurant_B_review_indicies:
    review_id = dfR.iloc[review_index].review_id
    cosine_df[review_id] = pd.Series(find_similar(reviews_tfidf,review_index), index=index)

final_scores = pd.Series()

final_scores.index.name = 'business_id'
for biz_id in businesses.index:
    cosines = cosine_df.query("business_id == '"+biz_id+"'")
    score = cosines.mean(axis=1).mean(axis=0)
    medians = cosines.median(axis=1).median(axis=0)
    final_scores.set_value(biz_id,score)


businesses['scores'] = final_scores
businesses['medians'] = medians
businesses.sort_values('scores',ascending=False)[::]
