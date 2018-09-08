import pandas as pd
import numpy as np
pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',80)

#load tips .csv file & remove extraneous columns
tips = pd.read_csv('data/yelp_academic_dataset_tip.csv')
tips = tips.drop(columns=['date','likes','user_id'])
tips = tips.rename(columns={'text': 'tip'})

#load existing Las Vegas restaurant file with all reviews
LV_restaurants = pd.read_pickle("./df_LVrestaurants.pkl")

LV = LV_restaurants.business_id.unique()  #list of unique Las Vegas business id's

#create df of tips in Las Vegas restaurants
tipsLV = tips[tips['business_id'].isin(LV)]

#MERGE LV_restaurants with tipsLV.... crashes :(
#df = pd.merge(LV_restaurants, tipsLV, on='business_id', how='inner')


#Create new dataframe for LV restaurants & tips instead of reviews:
#filter business dataframe for Las Vegas restaurants, remove extra columns
business = pd.read_csv('data/yelp_academic_dataset_business.csv')
business['categories'] = business['categories'].fillna("None")
business = business[business['categories'].str.contains('Restaurant')]
business = business[business['city'] == "Las Vegas"]
business = business.drop(columns=['attributes','hours','is_open','attributes_AcceptsInsurance',\
                      'attributes_AgesAllowed','attributes_BusinessAcceptsBitcoin','attributes_DogsAllowed',\
                      'attributes_HairSpecializesIn','attributes_CoatCheck','attributes_BusinessParking',\
                      'attributes_BikeParking','attributes_ByAppointmentOnly'])

tipsLV = pd.merge(business, tipsLV, on='business_id', how='inner')
tipsLV = tipsLV[tipsLV['stars'] >= 3]  #keep restaurants w/ 3+ stars
tipsLV = tipsLV[tipsLV['review_count'] >= 25]  #keep restaurants w/ > 25 reviews
tipsLV.to_pickle("./df_LVtips.pkl")
