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
import pickle
#%%

#%%
pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',50)
#%%

#------Load dataframe of restaurant reviews for restaurants with >25 reviews

df = pd.read_pickle("./df_LVrestaurants25samples.pkl")

#sample 25 reviews from each restaurant
#df = df1.groupby('business_id').apply(lambda x: x.sample(25))

#-------Create smaller dataframe (fewer columns) for simplified viewing and save the review_ids!
dfR = df.filter(["name","address","business_id","review_id","text", "stars_x"])
dfR = dfR.set_index('review_id')
dfR = dfR.reset_index()
review_ids=list(dfR.review_id)


#-------Spacy

nlp  = spacy.load('en_core_web_md')
type(nlp)
#Lemmatise
def keep_token(t):
    return (t.is_alpha and
            not (t.is_space or t.is_punct or
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if keep_token(t)]

docs = [lemmatize_doc(nlp(doc)) for doc in dfR.text]

pickle.dump(docs, open("docs_lemmatize.pkl", "wb"))

#Create a dictionary and filter stop and infrequent words
docs_dict = Dictionary(docs)
docs_dict.filter_extremes(no_below=5, no_above=0.5)
docs_dict.compactify()

#Bag of words for each documents, build TFIDF for each model and compute TF-IDF vector for each document
docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
docs_tfidf  = model_tfidf[docs_corpus]
docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
docs_vecs.shape

#Use Spacy to get the 300 dimentional  Globe embedding vector for each TF-IDF term
tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
tfidf_emb_vecs.shape

#Get a TF-IDF weighted Glove vector summary of each document
docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)
docs_emb.shape
pickle.dump(docs_emb, open("docs_emb.pkl", "wb"))

#PCA to reduce dimmentionality
docs_pca = PCA(n_components=8).fit_transform(docs_emb)
docs_pca.shape

pickle.dump(docs_pca, open("vectors_8PC.pkl", "wb"))

# Use t-sne to project the vectors to 2D.
tsne = manifold.TSNE()
viz = tsne.fit_transform(docs_pca)
vectors_2D = pd.DataFrame({'review_id':review_ids,'feat_1':viz[:,0], 'feat_2':viz[:,1]})

pickle.dump(vectors_2D, open("vectors_2D.pkl", "wb"))
pd.read_pickle("vectors_2D.pkl")

#%%

fig, ax = plt.subplots()
ax.margins(0.05)

bad_indices = np.where(dfR.stars_x < 2)[0]
good_indices = np.where(dfR.stars_x > 4)[0]

ax.plot(viz[bad_indices,0], viz[bad_indices,1], marker='o', linestyle='',
        ms=1, alpha=0.6)
ax.plot(viz[good_indices,0], viz[good_indices,1], marker='o', linestyle='',
        ms=1, alpha=0.3)

ax.legend()

plt.savefig('TSNE.png')
plt.show()
#%%
