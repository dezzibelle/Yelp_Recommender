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


class Recommender(object):
    """docstring for Recommender"""
    def __init__(self, state=None, city=None, min_review_count=20, is_open=1):
        super(Recommender, self).__init__()
        self.config = {
            'files':{
                'business':'../input/yelp_academic_dataset_business.json',
                'reviews':'../input/LVdf_restaurants.pkl',
                'users':'../input/yelp_academic_dataset_user.json',
                'tips':'../input/yelp_academic_dataset_tip.json',
                'checkin':'../input/yelp_academic_dataset_checkin.json'
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
        self.review_data = pd.read_pickle(self.config['files']['reviews'])
        self.review_data.rename(columns={'stars_x':'overall_stars','stars_y':'review_stars'}, inplace=True)
        self.review_data = self.review_data.dropna(subset=['user_id'])

        split_idx = int(self.review_data.shape[0]*0.8)
        train = self.review_data[['user_id','business_id','review_stars']].iloc[:split_idx,:]
        test = self.review_data[['user_id','business_id','review_stars']].iloc[split_idx:,:]
        self.train_data = tc.SFrame(train)
        self.test_data = tc.SFrame(test)
        print('review data read')

        arr = []
        with open(self.config['files']['business'], 'r') as input_file: 
            for line in input_file:
                arr.append(json.loads(line))

        business_arr = [self.parse_business(obj) for obj in arr]
        business_df = pd.DataFrame(business_arr)
        business_df = business_df[business_df.business_id.isin(self.review_data.business_id.unique())]

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

        self.business_data = pd.get_dummies(business_df[features_to_keep],
                                columns=[col for col in features_to_keep if col not in ['business_id',
                                                                                   'stars','review_count','is_open']])

        print('business info read')

        arr = []
        with open(self.config['files']['users'], 'r') as input_file: 
            for line in input_file:
                arr.append(json.loads(line))

        user_df = pd.DataFrame(arr)
        user_df = user_df[user_df.user_id.isin(self.review_data.user_id.unique())]
        user_df['yelping_years'] = (pd.to_datetime(datetime.now()) - pd.to_datetime(user_df['yelping_since'])).dt.days/365.0
        user_df.drop(['yelping_since','name'], axis=1, inplace=True)

        self.user_data = tc.SFrame(user_df)
        print('user info read')
        print('loading complete')

    def parse_business(self, business_obj):
        final_obj = {}
        # get first layer of attributes
        first_layer = ['business_id','stars','review_count','is_open']
        for a in first_layer:
            final_obj[a] = business_obj[a]
       
        if business_obj['attributes']:
            final_obj = {**final_obj, **business_obj['attributes']}
        if business_obj['hours']:
            final_obj = {**final_obj, **business_obj['hours']}
        try:
            if final_obj['BusinessParking']:
                parking = final_obj.pop('BusinessParking')
                parking = ast.literal_eval(parking)
                final_obj = {**final_obj, **parking}
        except:
        	pass
        if 'Ambience' in final_obj.keys():
            ambience = final_obj.pop('Ambience')
            ambience = ast.literal_eval(ambience)
            final_obj = {**final_obj, **ambience}
               
        if 'GoodForMeal' in final_obj.keys():
            meal = final_obj.pop('GoodForMeal')
            meal = ast.literal_eval(meal)
            final_obj = {**final_obj, **meal}
           
        return final_obj

    def train(self):
        new_train = self.train[self.train.user_id.isin(self.train.user_id.value_counts()[(self.train.user_id.value_counts()>15)].index)]
        new_train_data = tc.SFrame(new_train)
        self.model = tc.factorization_recommender.create(item_data=self.business_data,item_id='business_id',\
                                                    observation_data=self.new_train_data,user_id='user_id', \
                                                       target='review_stars',user_data=self.user_data)

    def evaluate_model(self):
        if self.model:
            self.model.evaluate_rmse(self.test_data, target='review_stars')

    def save_model(self):
        if model:
            pickle.dump(self.model, open(config['model']['model_path'], 'wb'))

    def load_model(self):
    	self.model = pickle.load(open(self.config['model']['model_path'], 'rb'))

    def predict(self, user_pair):
        user_pair = ['MtE3xl8AUYPbGWQQhY5IVQ','eyj4r8be__c7fVtfxeHr8Q']
        rec1 = factorization_model.recommend([user_pair[0]],100).to_dataframe()
        rec2 = factorization_model.recommend([user_pair[1]],100).to_dataframe()

        possible_recs = rec1.merge(rec2,on='business_id')
        possible_recs['total_score'] = possible_recs['rank_x'] + possible_recs['rank_y']

        return possible_recs[possible_recs.total_score == possible_recs.total_score.min()]['business_id'].iloc[0]

        

