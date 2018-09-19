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
from nltk.corpus import stopwords
#GenSim Doc2Vec libraries:
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import multiprocessing as mp
import os
import pickle
import multiprocessing
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
df
#sample 25 reviews from each restaurant
#df = df1.groupby('business_id').apply(lambda x: x.sample(25))

#-------Create smaller dataframe (fewer columns) for simplified viewing
dfR = df.filter(["name","address","business_id","review_id","text", "stars", "user_id"])
dfR = dfR.set_index('review_id')
dfR = dfR.reset_index()

#-------Creating a businesses dataframe
column_names=['name', 'address']
sample_name=dfR.groupby('business_id')[column_names].agg({"name": lambda x: x.unique(), "address":lambda x: x.unique()})
businesses=pd.DataFrame(sample_name)

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
#-------------------------------------------------------------------------------
results = pd.merge(RA_sim,RB_sim,on=('business_id','name','address'))
results['AB_sim'] = results[[('A_sim','median'),('B_sim','median')]].mean(axis=1)
results = results.sort_values(by="AB_sim",ascending=False)
#------------------------------------------------------------------------------------------
### Dataframe with statistics for most similar restaurants to BOTH restaurant A & restaurant B
results.head()


#-------------------------------------------------------------------------------
#----- TFIDF of all reviews using Gensim and Spacy
#-------------------------------------------------------------------------------
nlp  = spacy.load('en_core_web_sm')

#Lemmatise
def keep_token(t):
    return (t.is_alpha and
            not (t.is_space or t.is_punct or
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = [lemmatize_doc(nlp(doc)) for doc in dfR.text]
docs.to_pickle("docs_lemmatized.pkl")

#lemmatize function takes long time read pkl object instead
docs = pd.read_pickle("./docs_lemmatized.pkl")

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

#-------------------------------------------------------------------------------
#-----TFIDF with nltk
#-------------------------------------------------------------------------------
# Cleanup the text column by removing whitespace and punctuation.
text= dfR['text'].apply(lambda x: re.sub('\s+', ' ', x))
text= dfR.text.str.replace('<.*?>',' ').str.replace('\n',' ')

# Tokenize the text into a new corp column
corpus = list(zip(dfR.business_id, text))

stop = stopwords.words('english')

def tokenize(pair):
    id, text = pair
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()
    stems = ''

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        elif token in stop: continue
        stems += stem.stem(token) + ' '

    return(id,stems)

with mp.Pool() as pool:
    tokenized_reviews = pool.map(tokenize, corpus)

pd.DataFrame(tokenized_reviews)[1].isnull().any()
ids, texts = zip(*tokenized_reviews)
corp = texts


# Vectorize the corpus and store in a new nltk_dict column
def vectorize(doc):
    features=defaultdict(int)
    for token in doc.split(' '):
        features[token] += 1
    return features

from collections import defaultdict

with mp.Pool() as pool:
    nltk_dict = pool.map(vectorize, corp)


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')

reviews_tfidf_2 = tf.fit_transform(list(corp))

#-------------------------------------------------------------------------------
#-----Cosine similarity based on TFIDF matrix
#-------------------------------------------------------------------------------

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

dfR.groupby("business_id").indices[str.format(Rest_A.business_id[0])]

# With TFIDF generated Gensim way

for review_index in np.append(restaurant_A_review_indicies,restaurant_B_review_indicies):
    review_id = dfR.iloc[review_index].review_id
    cosine_df[review_id] = pd.Series(find_similar(reviews_tfidf,review_index), index=index)

    final_scores = pd.Series()

final_scores.index.name = 'business_id'
for biz_id in businesses.index:
    cosines = cosine_df.query("business_id == '"+biz_id+"'")
    score = cosines.mean(axis=1).mean(axis=0)
    medians = cosines.median(axis=1).median(axis=0)
    final_scores.set_value(biz_id,score)

business_g=businesses.copy(deep=True)
business_g['scores'] = final_scores
business_g['medians'] = medians

business_g.sort_values('scores',ascending=False)[::]

# With TFIDF matrix generated with nltk

for review_index in np.append(restaurant_A_review_indicies,restaurant_B_review_indicies):
    review_id = dfR.iloc[review_index].review_id
    cosine_df[review_id] = pd.Series(find_similar(reviews_tfidf_2,review_index), index=index)

final_scores = pd.Series()

final_scores.index.name = 'business_id'
for biz_id in businesses.index:
    cosines = cosine_df.query("business_id == '"+biz_id+"'")
    score = cosines.mean(axis=1).mean(axis=0)
    medians = cosines.median(axis=1).median(axis=0)
    final_scores.set_value(biz_id,score)

business_n = businesses.copy(deep=True)
business_n['scores'] = final_scores
business_n['medians'] = medians

business_n.sort_values('scores',ascending=False)[::]


#-------------------------------------------------------------------------------
#------------------------Model evaluation---------------------------------------
#-------------------------------------------------------------------------------
cf_output = pd.read_pickle("./res_sample_df.pkl")
cf_output.columns

list(dfR.columns)
#compute user-level review variance
review_variance = df.filter(["business_id", "user_id", "stars"])
review_variance = review_variance[["user_id", "stars"]].groupby("user_id").var()
review_variance = review_variance.fillna(0).reset_index()
review_variance = review_variance.rename(columns = {"stars": "review_variance"})
sns.distplot(review_variance.review_variance)

#make sample of restaurant pairs
business_ids = pd.DataFrame(df.business_id.unique())
rng = np.random.RandomState(1234)
sample_ids = rng.randint(1, business_ids.shape[0], 2000)
sample = business_ids.iloc[sample_ids]
sample.columns = ["business_id"]
sample.reset_index().drop("index", axis = 1)
sample.to_pickle("./yelp_dataset/sample_ids.pkl")

list1 = list(sample[:1000]["business_id"])
list2 = list(sample[1000:]["business_id"])
sample_business = pd.DataFrame({"business1_id": list1,
                                "business2_id": list2})

list1 = list(cf_output.business_id.unique()[:400])
list2 = list(cf_output.business_id.unique()[400:800])
sample_id_list = list1 + list2
len(sample_id_list) == 800
sample_business = pd.DataFrame({"business1_id": list1,
                                "business2_id": list2})

cf_output = cf_output[cf_output.business_id.isin(sample_id_list)]

#------------------------define functions---------------------------------------
#get user average rating
def get_user_avgscore(cf_output):
    user_avgscore = cf_output.groupby("user_id")["score"].mean()
    user_avgscore = user_avgscore.reset_index()
    user_avgscore.rename(columns = {"score": "avg_score"})

    return user_avgscore

#demean user ratings
def demean_cf_output(cf_output, user_avgscore):
    cf_output_demeaned = pd.merge(cf_output, user_avgscore, how = "inner", on = "user_id")
    cf_output_demeaned["score_dm"] = cf_output_demeaned.score - cf_output_demeaned.avg_score

    return cf_output_demeaned

#identify comparable restaurants
def identify_topcomps(restA, restB, cf_output):

    #compute user avg score
    user_avgscore = cf_output.groupby("user_id")["score"].mean()
    user_avgscore = user_avgscore.reset_index()
    user_avgscore =  user_avgscore.rename(columns = {"score": "avg_score"})

    #demean scores
    cf_output_demeaned = pd.merge(cf_output, user_avgscore, how = "inner", on = "user_id")
    cf_output_demeaned["score_dm"] = cf_output_demeaned.score - cf_output_demeaned.avg_score

    #select favorite restaurants
    df_favorites = cf_output_demeaned.query("rank <= 200")
    df_favorites = df_favorites.filter(["user_id", "business_id"])

    df_subA = cf_output_demeaned.query("business_id == '"+restA+"'") \
                    .sort_values("score_dm", ascending = False).iloc[:25, :] \
                    .filter(["user_id", "business_id", "score_dm"])
    comp_reviewersA = list(df_subA["user_id"])

    df_subB = cf_output_demeaned.query("business_id == '"+restB+"'") \
                    .sort_values("score", ascending = False).iloc[:25] \
                    .filter(["user_id", "business_id", "score_dm"])
    comp_reviewersB = list(df_subB["user_id"])

    set(comp_reviewersA).intersection(set(comp_reviewersB)) == set([])

    #identify top restaurants for each user
    cf_output_demeaned["top_600"] = cf_output_demeaned["rank"] <= 600
    top_restaurants = cf_output_demeaned.filter(["user_id", "business_id", "top_600"])

    top_restaurants_compA = top_restaurants[top_restaurants.user_id \
                                                            .isin(comp_reviewersA)]
    top_restaurants_compA["comp_set"] = "restA"

    top_restaurants_compB = top_restaurants[top_restaurants.user_id \
                                                            .isin(comp_reviewersB)]
    top_restaurants_compB["comp_set"] = "restB"

    return top_restaurants_compA, top_restaurants_compB

#output recommendation list given input choices
#c,
def get_recommendations(restA, restB, model = reviews_tfidf_2):

    rest1_review_ids = dfR[dfR.business_id==str.format(restA)].review_id.tolist()
    restaurant_A_review_indicies = dfR.groupby("business_id").indices[str.format(restA)]
    restaurant_B_review_indicies = dfR.groupby("business_id").indices[str.format(restB)]

    index = pd.MultiIndex.from_tuples(zip(dfR.business_id, dfR.review_id), \
                                            names=['business_id', 'review_id'])
    cosine_df = pd.DataFrame()
    cosine_df = pd.DataFrame(index=index)

    dfR.groupby("business_id").indices[str.format(restA)]

    # With TFIDF generated Gensim way

    for review_index in np.append(restaurant_A_review_indicies,restaurant_B_review_indicies):
        review_id = dfR.iloc[review_index].review_id
        cosine_df[review_id] = pd.Series(find_similar(model,review_index), index=index)

    cosine_df = cosine_df.mean(axis = 1)
    cosine_df = cosine_df.reset_index()
    cosine_df.columns = ["business_id", "review_id", "avg_review_siml"]
    cosine_df = pd.merge(cosine_df, id_review_variance, \
                                            how = "inner", on = "review_id")
    cosine_df["wgt_avg_review_sml"] = cosine_df.avg_review_siml * \
                                            cosine_df.review_variance
    #cosine_df.head()

    final_scores_nowgt = cosine_df.groupby("business_id")["avg_review_siml"] \
                                        .agg(["mean", "median"]).reset_index()
    final_scores_nowgt = final_scores_nowgt.rename(columns = \
            {"mean": "avg_similarity_nowgt", "median": "median_similiarity_nowgt"})
    #final_scores_nowgt.head()

    final_scores_wgt = cosine_df.groupby("business_id")["wgt_avg_review_sml"] \
                                    .agg(["mean", "median"]).reset_index()
    final_scores_wgt = final_scores_wgt.rename(columns = \
                {"mean": "avg_similarity_wgt", "median": "median_similiarity_wgt"})

    final_scores = pd.merge(final_scores_nowgt, final_scores_wgt, \
                                    how = "inner", on = "business_id")
    final_scores.shape
    #final_scores.head()

    return final_scores

#evaluate recommendation list
def evaluate(recommendations, agg_method, sample_obs, top_comps_restA, top_comps_restB):

    recommendation = recommendations.filter(["business_id", agg_method])
    recommendation = recommendation.sort_values(agg_method, ascending = False)
    recommendation = recommendation.reset_index().drop("index", axis = 1)
    recommendation = recommendation.head(20)
    recommendation.shape
    recommendation.head()

    recommendation_ids = recommendation.filter(["business_id"])
    recommendation_ids.head()

    rec_qualityA = pd.merge(top_comps_restA, recommendation_ids, \
                                            how = "inner", on = "business_id")
    rec_qualityB = pd.merge(top_comps_restB, recommendation_ids, \
                                            how = "inner", on = "business_id")
    rec_qualityA.shape
    rec_qualityB.shape

    common_set = set(rec_qualityA.business_id.unique()).intersection(\
                                        set(rec_qualityB.business_id.unique()))

    rec_qualityA.head()
    rec_qualityB.head()

    probA_likes = rec_qualityA.top_600.sum()/rec_qualityA.shape[0]
    probB_likes = rec_qualityB.top_600.sum()/rec_qualityB.shape[0]
    prob_likes = probA_likes*probB_likes

    weight = agg_method.split("_")[0]

    cv_df = pd.DataFrame({"model": "tfidf_nltk",
                          "aggregation": agg_method,
                          "weighting": weight,
                          "probability": [prob_likes],
                          "observation": sample_obs})

    return cv_df
#-------------------------------------------------------------------------------

restA = sample_business.iloc[0]["business1_id"]
restB = sample_business.iloc[0]["business2_id"]

df_topcomps_restA, df_topcomps_restB = identify_topcomps(restA, restB, cf_output)
df_topcomps_restA.head()

#compute user-level weights
unq_review_variance = dfR[["user_id", "business_id", "stars"]]
unq_review_variance = unq_review_variance[["user_id", "stars"]].groupby("user_id").var()
unq_review_variance = unq_review_variance.fillna(0).reset_index()
unq_review_variance = unq_review_variance.rename(columns = {"stars": "review_variance"})

#create ids for mapping weights back to vectorization output
id_review_variance = pd.merge(dfR, unq_review_variance, how = "inner", \
                                    on = "user_id")
id_review_variance = id_review_variance[["review_id", "review_variance"]]

#----------------------generate evaluation output-------------------------------
sample_size = 50
#agg_method = ["avg_similarity_nowgt", "median_similiarity_nowgt", \
#                        "avg_similarity_wgt", "median_similiarity_wgt"]
agg_method = ["avg_similarity_wgt"]
cv_output = pd.DataFrame()
for i in range(sample_size):

    print("Starting loop: ", i)

    restA = sample_business.iloc[i]["business1_id"]
    restB = sample_business.iloc[i]["business2_id"]

    df_topcomps_restA, df_topcomps_restB = identify_topcomps(restA, restB, cf_output)

    recs = get_recommendations(restA, restB)

    for l in range(len(agg_method)):

        print("Starting aggregation method: ", agg_method[l])
        cv_output = cv_output.append(evaluate(recs, agg_method[l], \
                                    i, df_topcomps_restA, df_topcomps_restB))

cv_output

#save evaluation output
cv_output.to_csv("./yelp_dataset/output/cv_output_nltk_tfidf.csv")
