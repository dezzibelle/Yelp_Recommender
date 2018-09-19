import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join
from datetime import datetime

import pickle
import ast
import json

import turicreate as tc

import matplotlib
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
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler

class Recommender(object):
    """docstring for Recommender"""
    def __init__(self):
        super(Recommender, self).__init__()
        self.config = {
            'files':{
                'business':'/Users/arianiherrera/Desktop/Ariani_yelp/Ariani/input/business_df.pkl',
                'reviews':'/Users/arianiherrera/Desktop/Ariani_yelp/Ariani/input/LVdf_restaurants.pkl',
                'users':'/Users/arianiherrera/Desktop/Ariani_yelp/Ariani/input/user_df.pkl',
                },
            'model':{
                   'model_path':'../output/model.pkl'
            }
        }
        self.review_data = None
        self.business_data = None
        self.user_data = None
        
        self.train_data = None
        self.test_data = None
        self.model = None
    def load_data(self):
        self.business_data = pd.read_pickle(self.config['files']['business'])
        print('business data read')
        self.user_data = pd.read_pickle(self.config['files']['users'])
        print('user data read')
        self.review_data = pd.read_pickle(self.config['files']['reviews'])
        self.review_data.rename(columns={'stars_x':'overall_stars','stars_y':'review_stars'}, inplace=True)
        self.review_data = self.review_data.dropna(subset=['user_id'])
        print('review data read') 
        print('loading complete')
        
    def prepare_model_input(self, use_text_flag=False):
        # prepare user matrix
        self.user_data['num_friends'] = self.user_data['friends'].apply(lambda x: len(x.split(',')))
        self.user_data['num_years_elite'] = self.user_data['elite'].apply(lambda x: len(x.split(',')))
        self.user_data.drop(['friends','elite'], axis=1, inplace=True)
        user_variance = self.review_data.groupby('user_id')['review_stars'].var().reset_index()
        user_variance.columns = ['user_id','rating_variance']
        self.user_data = self.user_data.merge(user_variance, how='left', on='user_id')
        self.user_data = self.user_data.fillna(0)
        
        user_scaler = StandardScaler()
        num_cols = [col for col in list(self.user_data) if col != 'user_id']
        self.user_data[num_cols] = user_scaler.fit_transform(self.user_data[num_cols])
        
        # prepare restaurant matrix
        features_to_keep = ['Alcohol','BikeParking','BusinessAcceptsCreditCards','Caters',
                   'GoodForKids','HappyHour','HasTV','NoiseLevel',
                   'OutdoorSeating','RestaurantsAttire','RestaurantsDelivery',
                   'RestaurantsGoodForGroups','RestaurantsPriceRange2',
                    'RestaurantsReservations','RestaurantsTableService',
                    'RestaurantsTakeOut','Smoking','WheelchairAccessible','WiFi','breakfast',
                    'brunch','business_id','casual','classy','dessert','dinner','divey',
                    'garage','hipster','intimate','is_open','latenight','lot','lunch',
                    'review_count','romantic','stars','street','touristy','trendy',
                    'upscale','valet','validated']
        self.business_data = pd.get_dummies(self.business_data[features_to_keep],
                                columns=[col for col in features_to_keep if col not in ['business_id',
                                                                                   'stars','review_count','is_open']])

        biz_scaler = StandardScaler()
        num_cols = [col for col in list(self.business_data) if col != 'business_id']
        self.business_data[num_cols] = biz_scaler.fit_transform(self.business_data[num_cols])
        
        if use_text_flag:
            Doc2Vec = pd.read_pickle('/Users/arianiherrera/Desktop/Ariani_yelp/Ariani/input/ReviewVectors3.pkl')
            vecs = Doc2Vec['vectors'].apply(pd.Series)
            vecs = vecs.rename(columns = lambda x : 'Vec_' + str(x))
            Doc = pd.concat([Doc2Vec[:], vecs[:]], axis=1)
            Doc = Doc.drop(["vectors"], axis=1)
            pca_Doc2Vec = PCA().fit(vecs)
            
            docs_pca = PCA(n_components=95).fit_transform(vecs)
            docs_pca = docs_pca.reshape(-1,95)
            docs_pca_df = pd.DataFrame(docs_pca)
            docs_pca_df.columns = ['text_pc_'+str(i) for i in range(1,96)]
            docs_pca_df = pd.merge(Doc2Vec, docs_pca_df, left_index=True, right_index=True)
            docs_pca_df_final = docs_pca_df.drop(columns=['vectors'])
            
            #self.review_data = self.review_data[['business_id','user_id','review_stars']].merge(docs_pca_df_final,
                                                                                               #on='review_id')
            columns_to_use = ['text_pc_'+str(i) for i in range(1,96)]
            items= ['business_id','user_id','review_stars','review_id']
            all_cols_to_use = items + columns_to_use
            self.review_data = self.review_data.merge(docs_pca_df_final, on='review_id')
            self.review_data = self.review_data[all_cols_to_use]
            vectors = pd.read_pickle('/Users/arianiherrera/Desktop/Ariani_yelp/Ariani/input/vectors_2D.pkl')
            self.review_data = self.review_data.merge(vectors, on ='review_id', how = 'inner')
            
            
            self.user_data = self.user_data[self.user_data.user_id.isin(self.review_data.user_id.unique())]
            self.business_data = self.business_data[self.business_data.business_id.isin(self.review_data.business_id.unique())]
        else:
            self.review_data = self.review_data[['business_id','user_id','review_stars']]
        # convert tables into SFrame for modeling
        self.user_data = tc.SFrame(self.user_data)
        self.business_data = tc.SFrame(self.business_data)
        
        split_idx = int(self.review_data.shape[0]*0.8)
        train = self.review_data.iloc[:split_idx,:]
        test = self.review_data.iloc[split_idx:,:]
        self.train_data = tc.SFrame(train)
        self.test_data = tc.SFrame(test)


    def train(self):
        self.model = tc.factorization_recommender.create(item_data=self.business_data,item_id='business_id',\
                                                    observation_data=self.train_data,user_id='user_id', \
                                                       target='review_stars',user_data=self.user_data)

    def evaluate_model(self):
        if self.model:
            return self.model.evaluate_rmse(self.test_data, target='review_stars')

    def save_model(self):
        if self.model:
            self.model.save('Users/arianiherrera/Desktop/Ariani_yelp/Ariani/output/model')
            #pickle.dump(self.model, open(config['model']['model_path'], 'wb'))

    def load_model(self):
        self.loaded_model = tc.load_model('Users/arianiherrera/Desktop/Ariani_yelp/Ariani/output/model')
        #self.model = pickle.load(open(self.config['model']['model_path'], 'rb'))

    def predict(self, user_pair):
        # user_pair = ['MtE3xl8AUYPbGWQQhY5IVQ','eyj4r8be__c7fVtfxeHr8Q']
        rec1 = factorization_model.recommend([user_pair[0]],100).to_dataframe()
        rec2 = factorization_model.recommend([user_pair[1]],100).to_dataframe()

        possible_recs = rec1.merge(rec2,on='business_id')
        possible_recs['total_score'] = possible_recs['rank_x'] + possible_recs['rank_y']

        return possible_recs[possible_recs.total_score == possible_recs.total_score.min()]['business_id'].iloc[0]
