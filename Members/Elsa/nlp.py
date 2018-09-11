#TEXT VECTORIZATION
import nltk
import string
import re
from collections import defaultdict
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.metrics.pairwise import linear_kernel

#Donwload sample
#sample= pd.read_csv('LVdf_minisample.csv')
#sample.sample(3)

#stopwords

stop = stopwords.words('english')

### Single threaded tokenizer
def tokenize_st(text):
  stem = nltk.stem.SnowballStemmer('english')
  text = text.lower()

  for token in nltk.word_tokenize(text):
    if token in string.punctuation: continue
    elif token in stop: continue
    yield stem.stem(token)

### Multithreaded tokenizer
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

# Assumes we're tokenized
def vectorize(doc):
    features=defaultdict(int)
    for token in doc.split(' '):
        features[token] += 1
    return features

def find_similar(tfidf_matrix, index, top_n = 10):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]
