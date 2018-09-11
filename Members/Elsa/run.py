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
sample=LVdf.sample(100000)
sample.columns.values.tolist()
sample.attributes_Ambience_romantic

# Cleanup the text column by removing whitespace and punctuation.
sample['text'] = sample['text'].apply(lambda x: re.sub('\s+', ' ', x))
sample['text'] = sample.text.str.replace('<.*?>',' ').str.replace('\n',' ')


# Tokenize the text into a new corp column
corpus = list(zip(sample.business_id, sample.text))

with mp.Pool() as pool:
    tokenized_reviews = pool.map(nlp.tokenize, corpus)

pd.DataFrame(tokenized_reviews)[1].isnull().any()
ids, texts = zip(*tokenized_reviews)
sample['corp'] = texts


# Group the "review" dataframe by business and join reviews into a single corpus
str_business = sample.groupby('business_id')['corp'].apply(lambda x: ' '.join(x))
str_business_df = pd.DataFrame(str_business)


# Vectorize the corpus and store in a new nltk_dict column
with mp.Pool() as pool:
    str_business_df['nltk_dict'] = pool.map(nlp.vectorize, str_business_df['corp'])


# Let's see what our DF looks like for four samples
str_business_df.sample(4)

# For later doc2vec implementation
#corpus=[TaggedDocument(words,  ['d{}'.format(idx)]) for idx, words in enumerate(str_business_df.corp.str.split())]
#model=Doc2Vec(corpus, size=100, min_count=5, window=5, min_count=20)
#print(model.docvecs[0])

#
# tf_matrix = LemVectorizer.transform(str_business_df.corp.sample(2)).toarray()
# print tf_matrix


#TFIDF and Cosine similarity
sample_name=sample.groupby('business_id')['name'].agg({"name": lambda x: x.unique(), "review_count": lambda x: x.count()})
sample_name
sample_name_df=pd.DataFrame(sample_name)
sample_name_df

#str_business_df2=pd.merge(str_business_df, sample_name_df, on="business_id", how="left")
str_business_df2
#str_business_df2.to_csv('df_all_tokenized.csv')

# Add two columns to filter

str_business_df2=pd.merge(str_business_df, sample_name_df, on="business_id", how="left")
str_business_df2.index
str_business_df2.columns
# Figure out how to merge
column_names=["attributes_GoodForKids", "attributes_Ambience_romantic", "attributes_RestaurantsPriceRange2", "stars_x"]
more_columns = sample.groupby('business_id')[column_names].agg({"attributes_GoodForKids": lambda x: x.unique(), "attributes_Ambience_romantic":lambda x: x.unique(), "attributes_RestaurantsPriceRange2":lambda x: x.unique(), "stars_x":lambda x: x.unique()})
more_columns
#,\
  #'attributes_DietaryRestrictions_vegetarian', 'attributes_DietaryRestrictions_vegan', \
  #'attributes_RestaurantsPriceRange2','stars_x']

str_business_df3=pd.merge(str_business_df2, more_columns, on="business_id", how="left")
sample.columns.values.tolist()
str_business_df3
len(str_business_df3)

#str_business_df2.columns.values.tolist()

# Pick two businesses and combine their reviews


#rest_1 = str_business_df3.iloc[46]
#rest_2= str_business_df3.corp[99]

joined_corp = str_business_df3.corp[46] + str_business_df3.corp[99]

data = {'name':['Me and You'], 'corp':[joined_corp], 'nltk_dict':[nlp.vectorize(joined_corp)], 'review_count':[2]}
data_df=pd.DataFrame(data)
random_reviews=pd.concat([data_df, str_business_df2])

#Term Frequency Inverse Document Frequency normalization to a sparse matrix of occurrence counts

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

#Learn vocabulary and idf, return term-document matrix

#{tfidf} (t,d,D)={tf} (t,d)* {idf} (t,D)
#idf is the It is the logarithmically scaled inverse fraction of the documents
#that contain the word, obtained by dividing the total number of documents by
#the number of documents containing the term

tfidf_matrix = tf.fit_transform(list(random_reviews.corp))
#with mp.Pool() as pool:
#    tfidf_matrix = pool.map(tf.fit_transform, list(random_reviews.corp))


random_reviews.iloc[0]
random_reviews

for index, score in nlp.find_similar(tfidf_matrix, 0):
       print(score, random_reviews.iloc[index])
