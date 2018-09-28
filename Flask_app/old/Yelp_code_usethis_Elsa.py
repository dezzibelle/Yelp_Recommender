from collections import defaultdict
import nltk
import string
import re
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def process_restaurants(choice1, choice2, zipcode):

  str_business_df3 = pd.read_csv('str_business_df3.csv')
  df1 = pd.read_csv('Flask_sample.csv')
  # filter original df by zipcode
  df1 = df1[df1['postal_code'] == zipcode] 
  # get list of business ids in that zipcode
  # filter df3 using that list of business ids
  business_list = list(df1['business_id'].unique())
  #list_zipcode = list(df1['postal_code'].unique()) 
  # save that to new df
  str_business_df3 = str_business_df3[str_business_df3['business_id'].isin(business_list)] 
  print (str_business_df3.shape)  
  stop = stopwords.words('english')

  def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
      if token in string.punctuation: continue
      elif token in stop: continue
      yield stem.stem(token)
      
  def vectorize(doc):
      features=defaultdict(int)
      for token in tokenize(doc):
          features[token] += 1
      return features
    
  def find_similar(tfidf_matrix, index, top_n = 6):
      cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
      related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
      return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

  joined_corp = str_business_df3.corp.loc[str_business_df3.name==choice1].to_string() + str_business_df3.corp.loc[str_business_df3.name==choice2].to_string()

  data = {'name':['Me and You'], 'corp':[joined_corp], 'nltk_dict':[vectorize(joined_corp)], 'review_count':[2]}
  data_df=pd.DataFrame(data)
  random_reviews=pd.concat([data_df, str_business_df3], sort = True)

  #Term Frequency Inverse Document Frequency normalization to a sparse matrix of occurrence counts

  tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

  #Learn vocabulary and idf, return term-document matrix

  tfidf_matrix = tf.fit_transform(list(random_reviews.corp))

  random_reviews.iloc[0]
  random_reviews

  df = []

  for index, score in find_similar(tfidf_matrix, 0):
         print(score, random_reviews.iloc[index])
         df.append(random_reviews.iloc[index])
  df = df1[df1['business_id'].isin([df[x]['business_id'] for x in range(0,6)])].reset_index()
  # arrange based on the score, drop duplicates, avoid dropping the inputs by default
  print (df.shape)
  return(df)

  