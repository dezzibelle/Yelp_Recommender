import pandas as pd
import numpy as np
import string, re, random
from random import sample
import itertools

pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',80)

#GenSim Doc2Vec libraries:
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import os
cores = multiprocessing.cpu_count()
from gensim.utils import simple_preprocess

def process_restaurants(choice1, choice2, zipcode):

    df = pd.read_csv("Flask_sample.csv")
    df_combo = df.filter(["name","business_id","text"],axis=1)
    #choice1 = "Gorilla Sushi"
    #choice2 = "Lazeez"


    ###-----DEFINE FUNCTIONS------###
    ###---------------------------###

    #-----Choose users' favorite restaurants:---#
    def user_restaurant(df,choice):
      #choice = df.sample() #choose random sample from df
      place = df[(df.name == choice)]  # <-- user input with specific restaurant business_id
      place = place.filter(["name","business_id","text"])
      #print("User's favorite restaurant is:",choice['name'].values[0],"\n")
      return(place)

    Rest_A = user_restaurant(df,choice1)
    print(Rest_A.name)
    Rest_B = user_restaurant(df,choice2)
    print(Rest_B.name)

    #---------Doc2Vec NLP---------#
    # MyDocs reading from df_combo data frame
    class MyDocs(object):
        def __iter__(self):
            for i in range(df_combo.shape[0]):
                yield TaggedDocument(words=simple_preprocess(df_combo.iloc[i,-1]), tags=['%s' % df_combo.iloc[i,0]])

    def doc2vec_nlp(df_combo, Rest_A, Rest_B, n=10):
        assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
        #combine Restaurants
        AB_textcombo = Rest_A['text'].values[0] + Rest_B['text'].values[0]
        AB_restaurants = [Rest_A['name'].values[0], Rest_B['name'].values[0]]

        #Add combined restaurants into one line & remove individual rows
        df_combo = df_combo.append({'name': AB_restaurants, 'business_id': '--', 'text': AB_textcombo}, ignore_index=True)
        df_combo = df_combo.drop(df_combo[(df_combo.business_id == Rest_A.business_id.values[0]) | (df_combo.business_id == Rest_B.business_id.values[0])].index)
        if not os.path.exists('models/doc2vec.model'):
            print("start training doc2vec model...")
            documents = MyDocs()
            doc2vec_model = Doc2Vec(dm=1, dbow_words=1, vector_size=200, window=8, min_count=20, workers=cores)
            doc2vec_model.build_vocab(documents)
            doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
            if not os.path.exists('models'):
                os.makedirs('models')
                doc2vec_model.save('models/doc2vec.model')
            else:
                doc2vec_model.save('models/doc2vec.model')
        else:
            doc2vec_model = Doc2Vec.load('models/doc2vec.model')
        results = pd.DataFrame(doc2vec_model.docvecs.most_similar(AB_restaurants, topn=n))
        results.columns = ['name', 'similarity']
        return(results)

    ##-----Load CSV file(s) with combined text (one row per restaurant)---#


    #--Run Doc2Vec NLP--#
    results = doc2vec_nlp(df, Rest_A, Rest_B, 10)
    df = results
    return(df)