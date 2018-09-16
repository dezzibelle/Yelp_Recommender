
import pandas as pd
import numpy as np
pd.set_option('max_columns',200)

business = pd.read_csv('data/yelp_academic_dataset_business.csv')
LVdf = business[business['city'] == "Las Vegas"]

review = pd.read_csv('data/yelp_academic_dataset_review.csv')
review.head()

LVdf = pd.merge(LVdf, review, on='business_id', how='inner')

del(review)

LVdf.to_pickle("./LVdf.pkl")  #all businesses

#  LVdf = pd.read_pickle("./LVdf.pkl")
#  LVdf_sample = pd.read_pickle("./LVdf_sample.pkl")

LVdf[LVdf['categories'].isna()]
LVdf['categories'] = LVdf['categories'].fillna("None")
LVdf = LVdf[LVdf['categories'].str.contains('Restaurant')]

LVdf.to_pickle("./LVdf_restaurants.pkl")
LVdf.sample(100000)).to_pickle("./LVdf_sample.pkl"
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',80)

df = LVdf[['address','business_id','categories','name','review_count','review_id','user_id','stars_y','text','useful']].copy()

#            'attributes_GoodForKids','attributes_RestaurantsGoodForGroups','attributes_GoodForMeal',\
#            'attributes_RestaurantsPriceRange2','attributes_DietaryRestrictions',\
#            'attributes_HappyHour','attributes_HasTV','attributes_Music','attributes_NoiseLevel',\
#            'attributes_GoodForDancing','attributes_DriveThru','attributes_Ambience','attributes_Alcohol',\
#            'attributes_Open24Hours','attributes_OutdoorSeating','attributes_RestaurantsAttire',\
#            'attributes_RestaurantsReservations','attributes_RestaurantsTableService','attributes_RestaurantsTakeOut',\
#            'review_count','stars_x','cool','date','funny','review_id','user_id','stars_y','text','useful')]

LVdf.sample(1000)).to_csv("./LVdf_minisample2.csv"



#Load csv files into pandas dataframe
#business = pd.read_csv('data/yelp_academic_dataset_business.csv')
#review = pd.read_csv('data/yelp_academic_dataset_review.csv')


tips = pd.read_csv('data/yelp_academic_dataset_tip.csv')

business['categories'] = business['categories'].fillna("None")

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
US_restaurants = business[business['state'].isin(states)][US_restaurants['categories'].str.contains('Restaurant')]
US_restaurants = US_restaurants[US_restaurants['categories'].str.contains('Restaurant')]

LV_restaurants = US_restaurants[US_restaurants['city'] == "Las Vegas"]

LV_restaurants = pd.merge(LV_restaurants, review, on='business_id', how='inner')
US_restaurants = pd.merge(US_restaurants, review, on='business_id', how='inner')

LV_restaurants = pd.merge(LV_restaurants, tips, on='business_id', how='inner')

LV_restaurants.to_pickle("./df_LVrestaurants.pkl")
US_restaurants.to_pickle("./df_USrestaurants.pkl")

del(business,review,tips)

#Filter Las Vegas restaurants & merge with corresponding reviews
df = business[business['city'] == "Las Vegas"]
df['categories'] = LVdf['categories'].fillna("None")
df = df[df['categories'].str.contains('Restaurant')]
df = pd.merge(df, review, on='business_id', how='inner')

#Save as pickle & .csv files
df.to_pickle("./LVdf_restaurants.pkl")
df.to_csv("./LVdf_restaurants.csv")

#Save samples
(df.sample(100000)).to_pickle("./LVdf_sample.pkl")
(df.sample(1000)).to_csv("./LVdf_minisample.csv")
