import pandas as pd
import numpy as np
import string, re, random
import itertools

pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',200)


#---------Load all LV restaurant reviews
df = pd.read_pickle("./data/LVdf_restaurants.pkl")  #Loading pd dataframe with ALL LasVegas restaurants reviews
df.shape
df.head(1)

df = df.drop(columns=['user_id','attributes','hours','attributes_AcceptsInsurance',\
                      'attributes_AgesAllowed','attributes_BusinessAcceptsBitcoin','attributes_DogsAllowed',\
                      'attributes_HairSpecializesIn','attributes_CoatCheck','attributes_BusinessParking',\
                      'attributes_BikeParking','attributes_ByAppointmentOnly'])
df['date'] = pd.to_datetime(df.date)
df.columns = df.columns.str.replace('attributes_','')
df.columns = df.columns.str.replace('Restaurants','')
df = df[df.is_open == 1]

df = df[df['review_count'] >= 25]  #keep restaurants w/ > 25 reviews
list(df) #show column names
df.shape   #874968 reviews total

df.to_pickle("./data/df_LVrestaurantsOpen25+.pkl")

df_revsample = df.groupby('business_id').apply(lambda x: x.sample(25))
df_revsample.shape
#df_revsample.to_pickle("./data/df_LVsampled25Reviews")
df_revsample.business_id.unique().shape  #  3020 unique businesses
df_revsample.name.unique().shape  #2163 business names (3020 - 2163 = 857 with multiple locations?)

#---------Repeat for Tips df: ###

df_tips = pd.read_pickle("./data/df_LVtips.pkl")  #Loading pd dataframe with ALL LasVegas restaurants tips
df_tips.shape
list(df_tips)
df_tips = df_tips.filter(['business_id','name','tip'])
df_tips.business_id.unique().shape
df_tips.to_pickle("./data/df_LVtips.pkl")

tip_count = df_tips.groupby(by="business_id").count()
tip_count = tip_count[tip_count['name']>=5]
tip_count.shape
