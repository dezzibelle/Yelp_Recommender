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

#read in yelp_academic_dataset_business
os.chdir("[INSERT DIRECTORY]")
review_sample = pd.read_csv("./yelp_dataset/review_sample.csv", error_bad_lines = False)
business_data = pd.read_json("./yelp_dataset/yelp_academic_dataset_business.json", lines = True)

business_data.columns
business_data.dtypes

business_data["categories"] = str(business_data["categories"])

business_data.count()

def is_restaurant(s):
    return s.lower().find("restaurant") != -1

business_data["categories"]

sum(list(map(is_restaurant, business_data.categories)))

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
sum(review_sample["id_length"] == 22) #identify actual ids

#subset data
cond1 = review_sample["text"] != "nan" #filter out missing reviews
cond2 = review_sample["id_length"] == 22 #filter out non-ids
df = review_sample[(cond1 == True) & (cond2 == True)]
df = df[["business_id", "text"]]

g = df.groupby("business_id").count()
g.describe()
g.sort_values(by = "text", ascending = False)

#subset business based on minimum no. of reviews
business_subset = pd.DataFrame(g[g["text"] > 20].index)
business_subset.head()
business_subset.shape

#restrict to business with more than 20 reviews in sample
df_subset = pd.merge(df, business_subset, how = "right", on = "business_id")
df_subset.shape
df.shape

#check review length before being aggregated
df_subset["review_len"] = df_subset.text.map(lambda x: len(x))
df_subset["review_len"].describe()

#combine individual reviews for each restaurant
df_subset_combinedrevs = df_subset.drop("review_len", axis = 1).groupby("business_id").sum()
df_subset_combinedrevs.head()

#check that review length reflects aggregation
df_subset_combinedrevs.head()
df_subset_combinedrevs.text.map(lambda x: len(x)).describe()

#plot distribution of aggregated review length
sns.distplot(df_subset_combinedrevs.text.map(lambda x: len(x)))

#generate sample input
random_ids = randint(1, len(df_subset_combinedrevs), 2) #random restaurant id
random_ids_text = df_subset_combinedrevs.iloc[random_ids]
random_ids_text["id"] = 1
random_ids_text

#merge sample input into single review
merged_review = random_ids_text.groupby("id").sum()
merged_review["business_id"] = "combined"
merged_review = merged_review.set_index("business_id")
merged_review

#check reviews aggregated
random_ids_text["text"].map(lambda x: len(x))
len(merged_review["text"].sum())

#combine merged review with sample
reviews_foranalysis = pd.concat([merged_review, df_subset_combinedrevs])
reviews_foranalysis.shape

#vectorize reviews
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(reviews_foranalysis["text"])

#view output of vectorization
vectors.shape
print(vectors)
print(vectorizer.vocabulary_)

count_similarity = cosine_similarity(vectors)
df_count_similarity = pd.DataFrame(count_similarity)

#compute tfidf from reviews
tfidf = TfidfVectorizer()
tfidfs = tfidf.fit_transform(reviews_foranalysis["text"])
tfidfs.shape
print(tfidfs)

%timeit tf_similarity = cosine_similarity(tfidfs)
df_tf_similarity = pd.DataFrame(tf_similarity)

#extract recommendation
recommendation = pd.DataFrame({})
recommendation["tfidf"] = tf_similarity[0, 1:]
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

#compute ratings matrix
reviews_sample_sub = review_sample.loc[:, ["business_id", "user_id", "stars"]]
reviews_sample_sub.describe()

#filter reviews data
review_sample["userid_length"] = review_sample["user_id"].str.len()
cond1 = review_sample["text"] != "nan" #filter out missing reviews
cond2 = review_sample["id_length"] == 22 #filter out non-ids
cond3 = review_sample["userid_length"] == 22 #filter out non-ids
cond4 = review_sample["stars"].isnull() == False #filter out missing ratings
cond5 = review_sample["user_id"].isnull() == False #filter out missing user ids
cond6 = review_sample["stars"].str.len() == 1
ratings_sample_sub = review_sample[(cond1 == True) & (cond2 == True)
                                 & (cond3 == True) & (cond4 == True)
                                 & (cond5 == True) & (cond6 == True)]
ratings_sample_sub = ratings_sample_sub.loc[:, ["business_id", "user_id", "stars"]]
ratings_sample_sub.head()

ratings_sample_sub["stars"] = ratings_sample_sub["stars"].astype(np.int64)
ratings_sample_sub.dtypes

#collapse ratings into unique rating per restaurant/user
unq_ratings_sample = ratings_sample_sub.groupby(["business_id", "user_id"])\
                        .mean().reset_index()
unq_ratings_sample.groupby(["business_id", "user_id"]).count()
unq_ratings_sample = unq_ratings_sample.assign(n_business_id = unq_ratings_sample\
                            .groupby("business_id").ngroup())
unq_ratings_sample = unq_ratings_sample.assign(n_user_id = unq_ratings_sample\
                            .groupby("user_id").ngroup()).sort_values(by = "n_user_id")
unq_ratings_sample.head()

#create ratings matrix
ratings_matrix = unq_ratings_sample.loc[:, ["n_business_id", "n_user_id", "stars"]]
ratings_matrix.head()

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

%timeit svd.fit(train)

#predict ratings
predictions = svd.test(test)
accuracy.rmse(predictions)

uid = str(100)
iid = str(200)
pred = svd.predict(uid, iid, verbose = True)
