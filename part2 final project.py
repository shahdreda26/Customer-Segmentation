#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import cophenet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score


# In[2]:


file_path = './E-commerce_data.xlsx' 
excel_data = pd.ExcelFile(file_path)

sheet_names = excel_data.sheet_names
print(sheet_names)


# In[3]:


customers=pd.read_excel("E-commerce_data.xlsx",sheet_name="customers")
genders=pd.read_excel("E-commerce_data.xlsx",sheet_name="genders")
cities=pd.read_excel("E-commerce_data.xlsx",sheet_name="cities")
transactions=pd.read_excel("E-commerce_data.xlsx",sheet_name="transactions")
branches=pd.read_excel("E-commerce_data.xlsx",sheet_name="branches")
merchants=pd.read_excel("E-commerce_data.xlsx",sheet_name="merchants")


# In[4]:


# Extract
customers.head()


# In[5]:


transactions.head()


# In[6]:


customers.tail()


# In[7]:


transactions.tail()


# In[8]:


customers.info()


# In[9]:


transactions.info()


# In[10]:


customers.describe(include='all')


# In[11]:


transactions.describe(include='all')


# In[12]:


# preprocessing
#datetime format
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
transactions['burn_date'] = pd.to_datetime(transactions['burn_date'])
customers['join_date'] = pd.to_datetime(customers['join_date'])


# In[13]:


# recency : day(last transaction)
recent_date = transactions['transaction_date'].max()
recency = transactions.groupby('customer_id')['transaction_date'].max().reset_index(name='last_transaction_date')
transactions['recency'] = (recent_date - recency['last_transaction_date']).dt.days
recency = transactions[['customer_id', 'recency']]
recency.head(10)


# In[14]:


city = customers[['customer_id', 'city_id']]
city


# In[15]:


gender = customers[['customer_id', 'gender_id']]
gender


# In[16]:


# frequency (total number transaction)
transaction_count = transactions.groupby('customer_id').size().reset_index(name='transaction_count')
transaction_count


# In[17]:


total_coupons_burnt = transactions[transactions['transaction_status'] == 'burned'].groupby('customer_id').size().reset_index(name='total_coupons_burnt')
total_coupons_burnt


# In[18]:


# Merge all features into the main DataFrame
customers_features = transactions.merge(recency, on='customer_id', how='left')
customers_features = customers_features.merge(transaction_count, on='customer_id', how='left')
customers_features = customers_features.merge(city, on='customer_id', how='inner')
customers_features = customers_features.merge(gender, on='customer_id', how='inner')
customers_features = customers_features.merge(total_coupons_burnt, on='customer_id', how='left')
customers_features


# In[19]:


customers_features.isnull().sum()


# In[20]:


customers_features.fillna(0, inplace=True)


# In[21]:


customers_features.isnull().sum()


# In[22]:


customers_features.duplicated().sum()


# In[23]:


customers_features = customers_features.drop_duplicates()


# In[24]:


customers_features.duplicated().sum()


# In[25]:


features = customers_features[['gender_id', 'city_id','recency_y','transaction_count','total_coupons_burnt']] 

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
features_encoded = encoder.fit_transform(features)

features_encoded_df = pd.DataFrame(features_encoded)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_encoded_df)



# In[26]:


#accuracy
distances = pdist(features_scaled)#Compute the pairwise distances between observations


# In[27]:


#Hierarchical clustering (Agglomerative (bottom-up))
linkage_methods = ['single', 'complete', 'average', 'ward']
cophenetic_correlation = {}
for method in linkage_methods:
    Z = linkage(features_scaled, method=method, metric='euclidean')
    c, _ = cophenet(Z, distances)# calculate the cophenetic correlation  between the linkage matrix Z and the distance matrix distances
    #cophenet: function returns two values 
    # c : the cophenetic correlation coefficient 
    # _ : (blank variable) cophenetic distance matrix is discarded
    cophenetic_correlation[method] = c 


# In[28]:


# Print the cophenetic correlation coefficient for each linkage method
for method, cophenetic_coef in cophenetic_correlation.items():
    print(f"Cophenetic Correlation Coefficient for {method} linkage method: {cophenetic_coef*100:.0f} %")


# In[31]:


n_clusters = 5
clusters = fcluster(Z, n_clusters, criterion='maxclust')
customers_features['segment'] = clusters


# In[32]:


silhouette_avg = silhouette_score(features_scaled, customers_features['segment'])
print(f'Silhouette Score: {silhouette_avg}')


# In[54]:


segment_analysis = customers_features.groupby('segment').agg({
    'transaction_count' : 'count',
    'total_coupons_burnt' : 'count',
    'customer_id': 'count',
    'gender_id': lambda x: x.mode()[0],  
    'city_id': lambda x: x.mode()[0]  
}).reset_index()

print(segment_analysis)


# In[55]:


# Visualizing Transaction Counts by Segment
plt.figure(figsize=(10, 6))
sns.barplot(data=segment_analysis, x='segment', y='transaction_count', palette='viridis')
plt.title('Transaction Count by Segment')
plt.xlabel('Segment')
plt.ylabel('Transaction Count')
plt.xticks(rotation=45)
plt.show()


# In[56]:


# Visualizing Total Coupons Burnt by Segment
plt.figure(figsize=(10, 6))
sns.barplot(data=segment_analysis, x='segment', y='total_coupons_burnt', palette='viridis')
plt.title('Total Coupons Burnt by Segment')
plt.xlabel('Segment')
plt.ylabel('Total Coupons Burnt')
plt.xticks(rotation=45)
plt.show()


# In[59]:


# Visualizing Gender Distribution by Segment
gender_map = {1: 'Male', 2: 'Female'}  # Replace with your actual values
segment_analysis['gender_id'] = segment_analysis['gender_id'].map(gender_map)

gender_counts = segment_analysis.groupby('gender_id')['segment'].count().reset_index()
plt.figure(figsize=(8, 6))
sns.countplot(data=segment_analysis, x='gender_id', palette='pastel')
plt.title('Gender Distribution Across Segments')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[58]:


# Visualizing City Distribution by Segment
city_counts = segment_analysis.groupby('city_id')['segment'].count().reset_index()
plt.figure(figsize=(12, 6))
sns.countplot(data=segment_analysis, x='city_id', palette='pastel')
plt.title('City Distribution Across Segments')
plt.xlabel('City')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




