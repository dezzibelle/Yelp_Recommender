import pandas as pd
import numpy as np
import string, re, random
import itertools
from random import sample
#GenSim Doc2Vec libraries:
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#Zipcode search libraries:
from math import radians, cos, sin, asin, sqrt, atan2, sqrt
#Install from terminal:  pip install uszipcode
from uszipcode import SearchEngine

pd.set_option('max_columns',200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows',50)


#-------Load sampled data frame with 25 reviews per restaurant
#df_rest = pd.read_pickle("newdf_LVrestaurants25samples.pkl")

def process_restaurants(df,choice1,choice2,user_zip = 89109,user_dist = 5):
    #-------Create smaller dataframe for doc2vec similarity algorithm
    dfR = df.filter(["name","neighborhood","address","business_id","review_id"])
    dfR = dfR.set_index('review_id')
    dfR = dfR.reset_index()
    df['address'] = df.address.fillna(' ')
    df['OPTIONS'] = df['name']+' ('+df['address']+')'
    doc2vec_model = Doc2Vec.load('models/doc2vec.model')

    #-----Create a dataframe for each user restaurant--------
    Rest_A = choice1
    Rest_B = choice2
    #creating dataframes with all of the users' chosen restaurants' reviews
    RA_df = df[df.OPTIONS == Rest_A].set_index('review_id').reset_index()
    RB_df = df[df.OPTIONS == Rest_B].set_index('review_id').reset_index()
    print("Choice1:", RA_df.name.values[1])
    print("Choice2:", RB_df.name.values[1])

    #----Find similarity to each restaurant----------------
    RA_sim = pd.DataFrame(doc2vec_model.docvecs.most_similar(RA_df.iloc[0:25,0], topn=len(dfR)))
    RA_sim.columns = ['review_id','A_sim']
    RA_sim = pd.merge(RA_sim,dfR,on="review_id",how="left")
    RA_sim = RA_sim.groupby(by=["business_id","name","neighborhood","address"]).agg(['mean','median','min','max']).sort_values(by=("A_sim","median"),ascending=False).reset_index()
    #print("Restaurants similar to A:\n",RA_sim.iloc[0:5,1])

    RB_sim = pd.DataFrame(doc2vec_model.docvecs.most_similar(RB_df.iloc[0:25,0], topn=len(dfR)))
    RB_sim.columns = ['review_id','B_sim']
    RB_sim = pd.merge(RB_sim,dfR,on="review_id",how="left")
    RB_sim = RB_sim.groupby(by=["business_id","name","neighborhood","address"]).agg(['mean','median','min','max']).sort_values(by=("B_sim","median"),ascending=False).reset_index()
    #print("Restaurants similar to B:\n",RB_sim.iloc[0:5,1])

    #-------------- Arrange & sort results
    results = pd.merge(RA_sim,RB_sim,on=('business_id','name',"neighborhood",'address'))
    results['AB_sim'] = results[[('A_sim','median'),('B_sim','median')]].mean(axis=1)
    results = results.sort_values(by="AB_sim",ascending=False)
    print("Restaurants similar to A&B:\n", results.iloc[0:5,1])

    #-------------- Organize & filter results dataframe
    results = results[["business_id","name","neighborhood","address","AB_sim"]]
    results.columns = results.columns.droplevel(-1)

    df = (df.filter(['business_id', 'name', 'neighborhood', 'latitude', 'longitude',
    'address','city','state', 'postal_code', 'review_count', 'stars',
    'categories', 'PriceRange2','Ambience_casual',
    'Ambience_classy', 'Ambience_divey', 'Ambience_hipster',
    'Ambience_intimate', 'Ambience_romantic', 'Ambience_touristy',
    'Ambience_trendy', 'Ambience_upscale', 'GoodForKids']))

    results = pd.merge(results, df,on=['business_id','name','neighborhood','address'],how="left")
    results = results.drop_duplicates()
    print("Restaurants similar to A&B after merging df and results:\n", results.iloc[0:5,1])

    def FilterPrice(df):
        if RA_df.PriceRange2[0] == RB_df.PriceRange2[0]:
            df = df[df.PriceRange2 == RA_df.PriceRange2[0]]
        elif RA_df.PriceRange2[0] > RB_df.PriceRange2[0]:
            df = df[(df.PriceRange2 <= RA_df.PriceRange2[0]) & (df.PriceRange2 >= RB_df.PriceRange2[0])]
        else:
            df = df[(df.PriceRange2 >= RA_df.PriceRange2[0]) & (df.PriceRange2 <= RB_df.PriceRange2[0])]
        return(df)

    results = FilterPrice(results)
    print("Restaurants similar to A&B after filtered by price:\n", results.iloc[0:5,1])
    print(user_zip)

#Filtering results based on the Zipcode

    search = SearchEngine(simple_zipcode=True)
    zipcode = search.by_zipcode(user_zip)
    user_lat = radians(zipcode.lat)
    user_lon = radians(zipcode.lng)

    def DistanceCalc(row):
        r_lat = radians(row.latitude)
        r_lon = radians(row.longitude)
        dlat = user_lat - r_lat
        dlon = user_lon - r_lon
        a = sin(dlat/2)**2 + cos(r_lat) * cos(user_lat) * sin(dlon/2)**2
        distance = 3960 * (2 * atan2(sqrt(a), sqrt(1 - a)))
        return(distance)

    results['Dist_mi'] = results.apply(DistanceCalc,axis=1)
    results = results[results['Dist_mi'] <= user_dist]

    #Add additional filter here for Kid-friendly & romantic/intimate/classy/upscale
    print(results.iloc[0:5,1])
    #print(results.head(5))
    return(results)
    