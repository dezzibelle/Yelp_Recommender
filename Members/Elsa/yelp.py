#%%
import json
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
#%%

def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.copy().items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

for json_filename in glob('*.json'):
    csv_filename = '%s.csv' % json_filename[:-5]
    print('Converting %s to %s' % (json_filename, csv_filename))
    df = pd.DataFrame([convert(line) for line in open(json_filename, encoding='utf-8')])
    df.to_csv(csv_filename, encoding='utf-8', index=False)

#Convert csv to pd df
review = pd.read_csv('yelp_academic_dataset_review.csv')
business = pd.read_csv('yelp_academic_dataset_business.csv')
checkin = pd.read_csv('yelp_academic_dataset_checkin.csv')
tip= pd.read_csv('yelp_academic_dataset_tip.csv')
user= pd.read_csv('yelp_academic_dataset_user.csv')

#AWS credential setup

# pip install awscli
# pip install boto3
# aws configure
# Configure guide: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
##Configure-QuickConfigure-IAMconsole-DeleteYourRootAccessKeys-ManageSecurityCredentials-AccessKeys-CreateNewKey

import boto3
s3 = boto3.resource('s3')
business = s3.Object("yelp-unsupervised-foodies", "yelp_academic_dataset_business.csv")\
    .get()["Body"].read().decode("utf-8")

for object in my_bucket.objects.all():
    key = object.key
    print( "Getting " + key )
    #body = object.get()['Body'].read()


#Business data exploration

business.shape #(188593, 61)
business.columns
business.sample(3)
business.isnull().sum().sort_values(ascending=False)
business.business_id.is_unique

business.city.value_counts
business.state.value_counts().plot(kind='bar',color='#008080', figsize=(12,6))
business.loc[business["state"]=="NV", "city"].unique()
business.loc[business["state"]=="NV", "city"].value_counts()

#Create a business df for Las Vegas
business_Vegas=business.loc[business["city"]=="Las Vegas", ]
business_Vegas.shape

#%%
#business_Vegas.stars.value_counts().plot(kind='bar')
plt.hist(business_Vegas.stars, color='#008080', edgecolor='white', bins=5)
plt.xlabel("Rating")
plt.ylabel("Count")
#%%

#Review data exploration
review.sample(2)
review.columns
review.shape
review.stars.value_counts().plot(kind="bar")


#Tip data exploration
tip.columns
tip.sample(2)

#User data exploration
user.columns
user.sample(2)

#%%
plt.scatter(user.review_count, user.average_stars, lw=0, alpha=.2, color='#008080')
plt.xlabel("Number of Reviews")
plt.ylabel("Average Stars")
#%%

#Checkin data exploration
checkin.columns
checkin.sample(2)

#Selecting a sample DataFrame
LVdf = business[business['city'] == "Las Vegas"]
LVdf["categories"] = LVdf["categories"].fillna("None")
LVdf = LVdf[LVdf["categories"].str.contains("Restaurant")]
LVdf = pd.merge(LVdf, review, on="business_id", how="inner")
sample=LVdf.sample(100000)

#TEXT VECTORIZATION
import nltk
import string
import re
pd.set_option('display.max_colwidth', -1)

#Donwload sample
sample= pd.read_csv('LVdf_minisample.csv')
sample.sample(3)

#stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
print(stop)

def tokenize(text):
  stem = nltk.stem.SnowballStemmer('english')
  text = text.lower()

  for token in nltk.word_tokenize(text):
    if token in string.punctuation: continue
    elif token in stop: continue
    yield stem.stem(token)




# Process review column
sample['text'] = sample['text'].apply(lambda x: re.sub('\s+', ' ', x))
sample['text'] = sample.text.str.replace('<.*?>',' ').str.replace('\n',' ')
sample['corp'] = sample['text'].apply(lambda x: ' '.join(tokenize(x)))
str_business=sample.groupby('business_id')['corp'].apply(lambda x: ' '.join(x))
str_business_df=pd.DataFrame(str_business)
str_business_df.iloc[2,]

#nltk
from collections import defaultdict

def vectorize(doc):
    features=defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features

vectorize(str_business_df.corp.iloc[2,])
str_business_df['nltk_dict'] = str_business_df['corp'].apply(lambda x: vectorize(x))
str_business_df.sample(4)

#Gensim
import gensim

str_business_df

id2word=gensim.corpora.Dictionary(str_business_df.corp.str.split())
vectors=[id2word.doc2bow(doc) for doc in str_business_df.corp.str.split()]
vectors

##Distributed representation
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

corpus=[TaggedDocument(words,  ['d{}'.format(idx)]) for idx, words in enumerate(str_business_df.corp.str.split())]
corpus
model=Doc2Vec(corpus, size=100, min_count=5, window=5, min_count=20)
print(model.docvecs[0])



import nltk
tf_matrix = LemVectorizer.transform(str_business_df.corp.sample(2)).toarray()
print tf_matrix
