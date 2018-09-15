#%%
import json
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
import re
import nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import spacy
pd.set_option('display.max_colwidth', -1)
#%%

# Load the converted csv files into pandas (both have been flattend three levels)
review = pd.read_csv('yelp_academic_dataset_review.csv')
business = pd.read_csv('yelp_academic_dataset_business.csv')
business.columns.values.tolist()

# Filter to Restaurants in Vegas
LVdf = business[business['city'] == "Las Vegas"]
LVdf["categories"] = LVdf["categories"].fillna("None")
LVdf = LVdf[LVdf["categories"].str.contains("Restaurant")]
LVdf = pd.merge(LVdf, review, on="business_id", how="inner")
#sample=LVdf
reviews=LVdf.sample(100000)
reviews.columns.values.tolist()
reviews.attributes_Ambience_romantic

# Cleanup the text column by removing whitespace and punctuation.
reviews['text'] = reviews['text'].apply(lambda x: re.sub('\s+', ' ', x))
reviews['text'] = reviews.text.str.replace('<.*?>',' ').str.replace('\n',' ')


# Tokenize the text into a new corp column
corpus = list(zip(reviews.business_id, reviews.text))

with mp.Pool() as pool:
    tokenized_reviews = pool.map(nlp.tokenize, corpus)

pd.DataFrame(tokenized_reviews)[1].isnull().any()
ids, texts = zip(*tokenized_reviews)
reviews['corp'] = texts
reviews.columns.tolist()

# Vectorize the corpus and store in a new nltk_dict column
with mp.Pool() as pool:
    reviews['nltk_dict'] = pool.map(nlp.vectorize, reviews['corp'])


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')

#Learn vocabulary and idf, return term-document matrix

#{tfidf} (t,d,D)={tf} (t,d)* {idf} (t,D)
#idf is the It is the logarithmically scaled inverse fraction of the documents
#that contain the word, obtained by dividing the total number of documents by
#the number of documents containing the term

#tfidf_matrix = tf.fit_transform(list(reviews.corp), y=list(reviews.review_id))


#with mp.Pool() as pool:
#    tfidf_matrix = pool.map(tf.fit_transform, list(random_reviews.corp))


import numpy as np
reviews_tfidf = tf.fit_transform(list(reviews.corp))
features = tf.get_feature_names()

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(reviews_tfidf, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(reviews_tfidf[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(reviews_tfidf, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids.any():
        D = reviews_tfidf[grp_ids].toarray()
    else:
        D = reviews_tfidf.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

top_feats_in_doc(reviews_tfidf, features, 0)

reviews.iloc[4].business_id
reviews.groupby("business_id").indices['KhWl4Mwhm_Oqq0zIIz-0wQ']
reviews.iloc[reviews.groupby("business_id").indices['EHDcxAlrrP4VPiZuQlrIrg']].name

reviews.groupby("business_id").indices['uuGL8diLlHfeUeFuod3F-w']
reviews_tfidf[4988].toarray()

top_mean_feats(reviews_tfidf, features, reviews.groupby("business_id").indices['uuGL8diLlHfeUeFuod3F-w'])


sample_name=reviews.groupby('business_id')['name'].agg({"name": lambda x: x.unique(), "review_count": lambda x: x.count()})
sample_name_df=pd.DataFrame(sample_name)

reviews.columns.tolist()
column_names=["attributes_GoodForKids", "attributes_Ambience_romantic", "attributes_RestaurantsPriceRange2", "stars_x", "latitude", "longitude", "postal_code"]
more_columns = reviews.groupby('business_id')[column_names].agg({"attributes_GoodForKids": lambda x: x.unique(), "attributes_Ambience_romantic":lambda x: x.unique(),\
 "attributes_RestaurantsPriceRange2":lambda x: x.unique(), "stars_x":lambda x: x.unique(), "latitude":lambda x: x.unique(), "longitude":lambda x: x.unique(),\
 "postal_code":lambda x: x.unique()})

more_columns
businesses=pd.merge(sample_name, more_columns, on="business_id", how="left")

#Filter reviews to 25 per business
reviews=reviews[reviews.review_count > 25]
reviews.groupby("business_id").indices['AjWFFZN8KwyqJzzi0DyEsw']

valid_indeces=[]
for index in reviews.groupby("business_id").indices:
    valid_indeces.append(np.random.choice(reviews.groupby("business_id").indices[index], size=25, replace=False, p=None))

reviews.iloc[valid_indeces[0]]


def find_similar(tfidf_matrix, review_index):
    cosine_similarities = cosine_similarity(tfidf_matrix[review_index:review_index+1], tfidf_matrix).flatten()
    return cosine_similarities

businesses.sample
#  Ichi Ramen Houset
businesses.loc['-ADtl9bLp8wNqYX1k3KuxA']

# Pasta Cucina
businesses.loc['zTkMh_RUVZW-J3lh7-l09Q']


rest1_review_ids = reviews[reviews.business_id=='-ADtl9bLp8wNqYX1k3KuxA'].review_id.tolist()

restaurant_1_review_indicies = reviews.groupby("business_id").indices['-ADtl9bLp8wNqYX1k3KuxA']
restaurant_2_review_indicies = reviews.groupby("business_id").indices['zTkMh_RUVZW-J3lh7-l09Q']


restaurant_1_review_indicies + restaurant_2_review_indicies

index = pd.MultiIndex.from_tuples(zip(reviews.business_id, reviews.review_id), names=['business_id', 'review_id'])
cosine_df = pd.DataFrame(index=index)

cosine_df
for review_index in restaurant_1_review_indicies:
    review_id = reviews.iloc[review_index].review_id
    cosine_df[review_id] = pd.Series(find_similar(reviews_tfidf,review_index), index=index)



final_scores = pd.Series()
final_scores.index.name = 'business_id'
for biz_id in businesses.index:
    score = cosine_df.query("business_id == '"+biz_id+"'").mean(axis=1).mean(axis=0)
    final_scores.set_value(biz_id,score)

businesses['scores'] = final_scores

businesses[businesses.stars_x>3].sort_values('scores',ascending=False)[:5:]
