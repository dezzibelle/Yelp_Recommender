import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import timeit
from sklearn.cluster import KMeans
sns.set()
from numpy.random import randint
import multiprocessing as mp

#read in yelp_academic_dataset_business
os.chdir("/Users/pcpu/Desktop/NYCData/Bootcamp/Projects/capstone")
review_sample = pd.read_csv("./yelp_dataset/review_sample.csv", error_bad_lines = False)
business_data = pd.read_json("./yelp_dataset/yelp_academic_dataset_business.json", lines = True)

business_data.columns
business_data.dtypes

business_data["categories"] = str(business_data["categories"])

business_data.count()

def is_restaurant(s):
    return s.lower().find("restaurant") != -1

business_data["categories"]

sum(list(map(is_restaurant, business_data["categories"])))

business_data["restaurant"] = list(map(is_restaurant, business_data["categories"]))

##review sample eda
review_sample.shape
review_sample.describe()

#list columns
review_sample.columns
review_sample.dtypes

#view data
review_sample.head(20)

#summarize business ids
review_sample["id_length"] = review_sample["business_id"].str.len()
sum(review_sample["id_length"] != 22) #identify actual ids

review_sample["userid_length"] = review_sample["user_id"].str.len()
sum(review_sample["userid_length"] != 22)

#subset data
cond1 = review_sample["text"] != "nan" #filter out missing reviews
cond2 = review_sample["id_length"] == 22 #filter out non-ids
cond3 = review_sample["userid_length"] == 22 #filter out non-ids
cond4 = review_sample["stars"].isnull() == False #filter out missing ratings
cond5 = review_sample["user_id"].isnull() == False #filter out missing user ids
cond6 = review_sample["stars"].str.len() == 1
df = review_sample[(cond1 == True) & (cond2 == True)
                                 & (cond3 == True) & (cond4 == True)
                                 & (cond5 == True) & (cond6 == True)]
df = df[["business_id", "user_id", "text", "stars"]]
df.head()

g = df.groupby("business_id").count()
g.describe()
g.sort_values(by = "text", ascending = False).head()

#subset business based on minimum no. of reviews
business_subset = list(g[g["text"] > 20].index)
business_subset

#compute user-level weights
df["stars"] = df["stars"].astype(np.int64)
unq_review_variance = df[df["business_id"].isin(business_subset)]
unq_review_variance.shape
unq_review_variance = df[["user_id", "business_id", "stars"]]
unq_review_variance = unq_review_variance[["user_id", "stars"]].groupby("user_id").var()
unq_review_variance = unq_review_variance.fillna(0).reset_index()
unq_review_variance = unq_review_variance.rename(columns = {"stars": "review_variance"})
unq_review_variance.head()
unq_review_variance.shape

#restrict to business with more than 20 reviews in sample
df_subset = df[df["business_id"].isin(business_subset)]
df_subset.head()
df_subset.shape
df.shape

#count of restaurants in sample
len(pd.Series(df_subset["business_id"].unique()))

#create ids for mapping weights back to vectorization output
id_review_variance = pd.merge(df_subset, unq_review_variance, how = "inner", \
                                    on = "user_id")
id_review_variance = id_review_variance[["business_id", "review_variance"]] \
                            .reset_index()
id_review_variance.head()
id_review_variance.shape

#define sparse matrix ids
sparse_matrix_ids = df_subset[["business_id", "user_id"]]\
                                .reset_index().drop("index", axis = 1)
sparse_matrix_ids.head()

#check review length before being aggregated
df_subset["review_len"] = df_subset["text"].str.len()
df_subset["review_len"].describe()
sns.distplot(df_subset["review_len"])

#vectorize reviews
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(df_subset["text"])

#view output of vectorization
type(vectors)
vectors.shape
print(vectors)
print(vectorizer.vocabulary_)

#convert vectorized reviews to coo format
vectors_coo = vectors.tocoo(copy = False)
print(vectors_coo)
vectors_df = pd.DataFrame({"row_no": vectors_coo.row,
                           "col_no": vectors_coo.col,
                           "word_count": vectors_coo.data})
vectors_df.shape
vectors_df.head()
vectors_df["row_no"].describe()
#compute weighted review count by restaurant
vectors_df["index"] = vectors_df["row_no"]
vectors_df = pd.merge(vectors_df, id_review_variance, how = "inner", on = "index")
vectors_df["weighted_count"] = vectors_df["word_count"] * vectors_df["review_variance"]

weighted_wordcount = pd.DataFrame(vectors_df.groupby(["business_id", "col_no"])\
                                    ["weighted_count"].sum()).reset_index()
weighted_wordcount = weighted_wordcount.assign(row_no = weighted_wordcount\
                                    .groupby("business_id").ngroup())
weighted_wordcount = weighted_wordcount.drop("business_id", axis = 1)
weighted_wordcount.head()

#compute cosine similarity using weighted word counts
from scipy.sparse import csr_matrix

row = np.array(weighted_wordcount["row_no"])
col = np.array(weighted_wordcount["col_no"])
data = np.array(weighted_wordcount["weighted_count"])
s_weighted_wordcount = csr_matrix((data, (row, col)))
count_similarity = cosine_similarity(s_weighted_wordcount)
df_count_similarity = pd.DataFrame(count_similarity)
df_count_similarity.shape
df_count_similarity.head()
#compute tfidf from reviews
tfidf = TfidfVectorizer()
tfidfs = tfidf.fit_transform(df_subset["text"])

#view output of vectorization
type(tfidfs)
tfidfs.shape
print(tfidfs)
print(tfidf.vocabulary_)

#convert vectorized reviews to coo format
tfidfs_coo = tfidfs.tocoo(copy = False)
print(tfidfs_coo)
tfidfs_df = pd.DataFrame({"row_no": tfidfs_coo.row,
                           "col_no": tfidfs_coo.col,
                           "tfidf": tfidfs_coo.data})
tfidfs_df.shape
tfidfs_df.head()
tfidfs_df["row_no"].describe()
#compute weighted review count by restaurant
tfidfs_df["index"] = tfidfs_df["row_no"]
tfidfs_df = pd.merge(tfidfs_df, id_review_variance, how = "inner", on = "index")
tfidfs_df["weighted_tfidf"] = tfidfs_df["tfidf"] * tfidfs_df["review_variance"]

weighted_tfidf = pd.DataFrame(tfidfs_df.groupby(["business_id", "col_no"])\
                                    ["weighted_tfidf"].sum()).reset_index()
weighted_tfidf = weighted_tfidf.assign(row_no = weighted_tfidf\
                                    .groupby("business_id").ngroup())
weighted_tfidf = weighted_tfidf.drop("business_id", axis = 1)
weighted_tfidf.head()

#compute cosine similarity using weighted word counts
row = np.array(weighted_tfidf["row_no"])
col = np.array(weighted_tfidf["col_no"])
data = np.array(weighted_tfidf["weighted_tfidf"])
s_weighted_tfidf = csr_matrix((data, (row, col)))
tfidf_similarity = cosine_similarity(s_weighted_tfidf)
df_tfidf_similarity = pd.DataFrame(tfidf_similarity)
df_tfidf_similarity.shape
df_tfidf_similarity.head()

#compute ratings matrix
ratings_sample_sub = df_subset.loc[:, ["business_id", "user_id", "stars"]]
ratings_sample_sub["stars"] = ratings_sample_sub["stars"].astype(np.int64)
ratings_sample_sub.dtypes

#check ratings are unique
ratings_sample_sub.groupby(["business_id", "user_id"]).count().describe()

ratings_sample_sub = ratings_sample_sub.assign(n_business_id = ratings_sample_sub \
                            .groupby("business_id").ngroup())
ratings_sample_sub = ratings_sample_sub.assign(n_user_id = ratings_sample_sub \
                            .groupby("user_id").ngroup()).sort_values(by = "n_user_id")
ratings_sample_sub.head()

#generate sample observations
random_ids = randint(1, len(df_subset), 2) #random restaurant id
random_ids_text = list(ratings_sample_sub.iloc[random_ids]["business_id"])
random_ids_text["id"] = 1
random_ids_text

ratings_sample_sub = ratings_sample_sub[ratings_sample_sub["business_id"] \
                                            .isin(random_ids_text)]

ratings_sample_sub[ratings_sample_sub["stars"] == 5]["user_id"]
#create ratings matrix
ratings_matrix = ratings_sample_sub.loc[:, ["n_business_id", "n_user_id", "stars"]]
ratings_matrix = ratings_matrix.rename(columns = {"n_business_id": "itemID", "n_user_id": "userID", \
                            "stars": "rating"})
ratings_matrix.head()
ratings_matrix["userID"].max()

#import ratings into surprise
from surprise import Reader
from surprise import Dataset

reader = Reader(rating_scale = (1, 5))
data = Dataset.load_from_df(ratings_matrix, reader)

#predict missing ratings
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.25)

svd = SVD()

svd.fit(train)

#predict ratings
predictions = svd.test(test)
accuracy.rmse(predictions)

uid = str(100)
uid
iid = str(2)
pred = svd.predict(uid, iid, verbose = True)
round(list(pred)[3], 0)

ratings_matrix["n_user_id"].describe()

pd.DataFrame(predictions).shape
#extract recommendation
recommendation["tfidf"] = df_tfidf_similarity[0, 1:]
recommendation["count_vector"] = count_similarity[0, 1:]
recommendation

#compare recommendation approaches visually
sns.jointplot("tfidf", "count_vector", data = recommendation)

#cluster restaurants
%timeit kmeans = KMeans(n_clusters = 5)

#cluster based on count vector output
%timeit kmeans.fit(vectors)
y_kmeans = kmeans.predict(vectors)
y_kmeans.shape
pd.Series(y_kmeans).value_counts().sort_index()

#cluster based on tfidf output
%timeit kmeans.fit(tfidfs)
y_kmeans_tfidf = kmeans.fit(tfidfs)
y_kmeans.shape
pd.Series(y_kmeans).value_counts().sort_index()

#check cosine similarity output
from sklearn.metrics.pairwise import cosine_similarity

arr1 = np.array([0, 1, 0, 0, 1]).reshape(1, -1)
arr2 = np.array([0, 0, 1, 1, 1]).reshape(1, -1)
arr3 = np.array([1, 1, 0, 1, 0]).reshape(1, -1)

cosine_similarity(arr1, arr2)
cosine_similarity(arr3, arr2)

from sklearn.metrics import pairwise_distances

A = np.array([[0, 1, 0, 0, 1],
              [0, 0, 1, 1, 1],
              [1, 1, 0, 1, 0]])

dist_out = 1-pairwise_distances(A, metric = "cosine")
dist_out
