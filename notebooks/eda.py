
# coding: utf-8

# # EDA

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
import implicit
import pickle
import os


# In[2]:


## Magic function lets the bigquery run in Jupyter Notebook
get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# ### Set connection with project and dataset made on GCP

# In[3]:


client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')


# In[4]:


def bq2df(sql):
    query = client.query(sql) # API request
    results = query.result()
    return results.to_dataframe()


# ### Raw data

# In[5]:


q = ('SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
    'LIMIT 100')
complete_data = bq2df(q)
complete_data.head()


# In[8]:


print("HITS :- is an array of the nested fields that are populated for any and all types of hits.\nTOTALS :- is the section contains aggregate values across the session.")


# ### Number of tables in dataset

# In[9]:


merch_data = client.get_dataset(merch_data_ref)
tables = [x.table_id for x in client.list_tables(merch_data)]
len(tables)


# The sample dataset contains jumbled Google Analytics 360 data from the Google Merchandise Store, a real ecommerce store for the period of 1-Aug-2016 to 1-Aug-2017.<br>
# It includes the following kinds of information:<br>
# 1. Traffic source data: information about where website visitors originate. This includes data about organic traffic, paid search traffic, display traffic, etc.<br>
# 2. Content data: information about the behavior of users on the site. This includes the URLs of pages that visitors look at, how they interact with content, etc.<br>
# 3. Transactional data: information about the transactions that occur on the Google Merchandise Store website.
# 
# <br>Orignal dataset contains total of 366 tables, one table for each day. Each row of the table represents a session and columns contains the detailed information about the session. Details on table schema and columns can be found here https://support.google.com/analytics/answer/3437719?hl=en

# ### Extracting Data for Recommender System

# In[10]:


q = """
    SELECT 
      CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING)) AS visitorId,
      hitNumber,
      time,
      page.pageTitle,
      type,
      productSKU,
      v2ProductName,
      v2ProductCategory,
      productPrice/1000000 as productPrice_USD

    FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`, 
      UNNEST(hits) AS hits,
      UNNEST(hits.product) AS hits_product

"""


# In[11]:


## Sample of data which we require 
dd = bq2df(q) 
dd.head()


# ### Loading Processed Data

# In[12]:


## Loading the processed data that I stored on GCP using processd_data.py
q = """
    SELECT *
    FROM merch_store.aggregate_web_stats
    """
df = bq2df(q)
df.head()


# In[13]:


print("------- DATA SUMMARY -------\n")
print("Number of rows in processed data : {}".format(df.shape[0]))
print("Number of columns in processed data : {}\n".format(df.shape[1]))
print("Number of Customers : {}".format(df['visitorId'].nunique()))
print("Total number of products : {}\n".format(df['itemId'].nunique()))
print("Stats of the processed data : \n{}\n".format(df.describe()))
print("Number of product details viewed by the users but at the end of the session = {}".format(sum(df['session_duration'] == 1)))


# ### Most Viewed Products

# Most viewed article_id, as well as how often it was viewed

# In[14]:


most_viewed_itemsId = df.groupby('itemId').count()
most_viewed_itemsId.sort_values('visitorId', ascending=False).head(10)
most_viewed_itemId_sorted = most_viewed_itemsId.sort_values('visitorId',ascending=False)
most_viewed_itemId_sorted.head(10)


# In[16]:


most_viewed_article = str(most_viewed_itemsId.index[0])
print('The most viewed item in the dataset : {}'.format(most_viewed_itemId_sorted.index[0]))
max_views = most_viewed_itemsId.values[0]
print('The most viewed item in the dataset was viewed how many times? {}'.format(most_viewed_itemId_sorted['visitorId'][0]))


# ### Histogram of Session Durations

# Session durations could be little weird depending on how pessimistic and optimistic the customer is. We can plot the histogram of session durations in the dataset:

# In[17]:


df['session_duration'].plot(kind= 'hist', logy=True, bins=100, figsize= [8,5])


# ### Normalize Session Duration

# So, let’s scale and clip it the values by the median session duration (average duration will be dramatically affected by the outliers):

# In[18]:


## Processed data is normalized and saved on GCP using normalized_data.py

norm_query = """
SELECT * 
FROM merch_store.recommendations_data
--WHERE 
--visitorId LIKE '6535875617262565784%';
"""
norm_df = bq2df(norm_query)
norm_df.head()


# In[19]:


## just testing if normalised data have same data as we have in processed data.
norm_df[norm_df['visitorId']== "0321971686985517255-1"]


# ### Histogram of Normalized Session Durations

# Now, the session duration is scaled and in the right range:

# In[20]:


norm_df['normalized_session_duration'].plot(kind= 'hist', logy=True, bins=100, figsize= [8,5])


# In[21]:


# dir_path = os.path.dirname(os.getcwd())
# norm_df.to_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))


# ### Understanding how this processed data was created

# <br>
# Check also for customer 0237983577378351965
# 

# In[22]:


get_ipython().run_cell_magic('bigquery', '', "SELECT\n    *\nFROM\n    merch_store.aggregate_web_stats\nWHERE \nvisitorId LIKE '6535875617262565784%';\n")


# In[23]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT CONCAT(fullVisitorID,\'-\',CAST(visitNumber AS STRING)) AS visitorId, hits,productSKU, time\nFROM \n    `dummy24571.merch_store.ga_sessions_2016*`,\nUNNEST(hits) AS hits,\nUNNEST(hits.product) AS hits_product\nWHERE\n    hits.hitNumber > 0 AND fullVisitorId = \'6535875617262565784\' AND eCommerceAction.action_type = "2"\nLIMIT 50')


# In[24]:


get_ipython().run_cell_magic('bigquery', '', "SELECT \n    CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING)) AS visitorId,\n    date, visitStartTime,\n    hits.hitNumber, totals.hits AS totals_hits,productSKU, hits.time, hits.hour\nFROM \n    `dummy24571.merch_store.ga_sessions_2016*`,\nUNNEST(hits) AS hits,\nUNNEST(hits.product) AS hits_product\nWHERE\n    hits.hitNumber > 0 AND fullVisitorId = '6535875617262565784' AND visitNumber = 2 AND hitNumber = 5\nLIMIT 95")


# In[25]:


# For customer 6535875617262565784-2, product = GGOEAKDH019899, time_duration is calculated by 
# taking the difference of click time of this product and the next product click time.
# this procuct hitNumber = 4 and time is 242108, next product click time is 292162.
hit4_time = 242108
hit5_time = 292162
sess_duration = hit5_time - hit4_time
sess_duration

