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
plt.hist(business_Vegas.stars, color='#008080', edgecolor='white')
plt.xlabel("Rating")
plt.ylabel("Count")
#%%

#Business data exploration
review.columns
review.shape
review.stars.value_counts()

review.stars.value_counts().plot(kind="bar")
