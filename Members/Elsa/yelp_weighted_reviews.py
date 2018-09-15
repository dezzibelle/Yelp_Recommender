#%%
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
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
reviews=LVdf.sample(1000)
review_ids = list(reviews.review_id)

# TBD: Change to a more appropriate dict/corp?
nlp  = spacy.load('en_core_web_md')

#Lemmatise
def keep_token(t):
    return (t.is_alpha and
            not (t.is_space or t.is_punct or
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = [lemmatize_doc(nlp(doc)) for doc in reviews.text]

#Create a dictionary and filter stop and infrequent words
docs_dict = Dictionary(docs)
docs_dict.filter_extremes(no_below=20, no_above=0.2)
docs_dict.compactify()

#Bag of words for each documents, build TFIDF for each model and compute TF-IDF vector for each document
docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
docs_tfidf  = model_tfidf[docs_corpus]
docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
docs_vecs.shape

#Use Spacy to get the 300 dimentional  Globe embedding vector for each TF-IDF term
tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
tfidf_emb_vecs

#Get a TF-IDF weighted Glove vector summary of each document
docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)
docs_emb.shape

#PCA to reduce dimmentionality
docs_pca = PCA(n_components=8).fit_transform(docs_emb)
docs_pca.shape

# Use t-sne to project the vectors to 2D.
tsne = manifold.TSNE()
viz = tsne.fit_transform(docs_pca)
viz
res = pd.DataFrame({'review_id':review_ids,'feat_1':viz[:,0], 'feat_2':viz[:,1]})
res
#%%

fig, ax = plt.subplots()
ax.margins(0.05)

reviews.columns
bad_indices = np.where(reviews.stars_x < 3)[0]
good_indices = np.where(reviews.stars_x >= 4)[0]

ax.plot(viz[bad_indices,0], viz[bad_indices,1], marker='o', linestyle='',
        ms=2, alpha=0.6)
ax.plot(viz[good_indices,0], viz[good_indices,1], marker='o', linestyle='',
        ms=2, alpha=0.3)

ax.legend()

plt.show()
#%%

def find_similar(tfidf_matrix, index, top_n = 10):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

for index, score in find_similar(viz, 10):
       print('Score:', score, ' index ', index, ' ', reviews.iloc[index].name)
