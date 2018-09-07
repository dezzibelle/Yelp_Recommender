import pandas as pd
import numpy as np
pd.set_option('max_columns',200)

#Load csv files into pandas dataframe
business = pd.read_csv('data/yelp_academic_dataset_business.csv')
review = pd.read_csv('data/yelp_academic_dataset_review.csv')

#Filter Las Vegas restaurants & merge with corresponding reviews
LVdf = business[business['city'] == "Las Vegas"]
LVdf['categories'] = LVdf['categories'].fillna("None")
LVdf = LVdf[LVdf['categories'].str.contains('Restaurant')]
LVdf = pd.merge(LVdf, review, on='business_id', how='inner')

#Save as pickle & .csv files
LVdf.to_pickle("./LVdf_restaurants.pkl")
LVdf.to_csv("./LVdf_restaurants.csv")

#Save samples
(LVdf.sample(100000)).to_pickle("./LVdf_sample.pkl")
(LVdf.sample(1000)).to_csv("./LVdf_minisample.csv")
