
import json
import pandas as pd
from glob import glob
import ast

##Convert json string to flat python dictionary

def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.copy().items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                try:
                    d = ast.literal_eval(vv)
                    for kkk, vvv in d.items():
                        ob['%s_%s_%s' % (k, kk, kkk)] = vvv
                    del ob[kk]
                except:
                    ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

def convert_json_in_current_directory():
    for json_filename in glob('*business.json'):
        csv_filename = '%s.csv' % json_filename[:-5]
        print('Converting %s to %s' % (json_filename, csv_filename))
        df = pd.DataFrame([convert(line) for line in open(json_filename, encoding='utf-8')])
        df.to_csv(csv_filename, encoding='utf-8', index=False)

#convert_json_in_current_directory()
#AWS credential setup

# pip install awscli
# pip install boto3
# aws configure
# Configure guide: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
##Configure-QuickConfigure-IAMconsole-DeleteYourRootAccessKeys-ManageSecurityCredentials-AccessKeys-CreateNewKey

import boto3
#Load from s3:
#s3 = boto3.resource('s3')
#business = s3.Object("yelp-unsupervised-foodies", "yelp_academic_dataset_business.csv")\
#    .get()["Body"].read().decode("utf-8")

def list_files_in_bucket(my_bucket):
    for object in my_bucket.objects.all():
        key = object.key
        print( "Getting " + key )
    #   body = object.get()['Body'].read()
