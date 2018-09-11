#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%

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
