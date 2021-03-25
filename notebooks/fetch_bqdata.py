
# coding: utf-8

# In[1]:


import pandas as pd
from google.cloud import bigquery
import pickle
import os


# In[2]:


client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')


# In[3]:


def bq2df(sql):
    query = client.query(sql) # API request
    results = query.result()
    return results.to_dataframe()


# In[4]:


q = ('SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
    'LIMIT 100')


# In[5]:


# bq2df(q)


# In[6]:


merch_data = client.get_dataset(merch_data_ref)


# In[7]:


merch_data


# In[8]:


# [x.table_id for x in client.list_tables(merch_data)]


# In[9]:


merch_full = client.get_table(merch_data.table('ga_sessions_20160801'))


# In[10]:


merch_full


# In[11]:


[command for command in dir(merch_full) if not command.startswith('_')]


# In[20]:


# merch_full.schema


# In[75]:


query = """
  SELECT
      fullVisitorId AS visitor_id,
      visitId AS visit_id,
      CONCAT(fullVisitorId, CAST(visitId AS STRING)) AS unique_session_id,
      date,
      product.v2ProductCategory AS product_category,
      product.v2ProductName AS product_name,
      product.productSKU AS product_sku,
      product.productPrice/1e6 AS product_price,
      product.productQuantity AS product_quantity,
      product.productRevenue/1e6 AS product_revenue,
      totals.totalTransactionRevenue/1e6 AS total_revenue
  FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`
      , UNNEST(hits) AS hits
      , UNNEST(hits.product) AS product
  WHERE
      _TABLE_SUFFIX BETWEEN '20170401' AND '20170430'
      AND geoNetwork.country = 'United States'
      AND productRevenue IS NOT NULL
  ORDER BY
      4 ASC, 10 DESC
"""


# In[91]:


data_0 = bq2df(query)


# In[93]:


data_0


# In[46]:


query = """
  SELECT
      fullVisitorId AS visitor_id,
      visitId AS visit_id,
      CONCAT(fullVisitorId, CAST(visitId AS STRING)) AS unique_session_id,
      date,
      product.v2ProductCategory AS product_category,
      product.v2ProductName AS product_name,
      product.productSKU AS product_sku,
      product.productPrice/1e6 AS product_price,
      totals.timeOnScreen AS time_spent,
      totals.transactions AS transection,
      product.productQuantity AS product_quantity,
      product.productRevenue/1e6 AS product_revenue,
  FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`
      , UNNEST(hits) AS hits
      , UNNEST(hits.product) AS product
  WHERE
      _TABLE_SUFFIX BETWEEN '20170401' AND '20170531'
      AND geoNetwork.country = 'United States'
      AND productRevenue IS NOT NULL
  ORDER BY
      4 ASC
"""


# In[47]:


data_1 = bq2df(query)


# In[48]:


data_1


# In[53]:


query = """
  SELECT
      CONCAT(fullVisitorId, CAST(visitId AS STRING)) AS unique_session_id,
      date,
      totals.totalTransactionRevenue/1e6 AS total_revenue
  FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  WHERE
      _TABLE_SUFFIX BETWEEN '20170401' AND '20170430'
      AND geoNetwork.country = 'United States'
      AND totals.totalTransactionRevenue IS NOT NULL
"""


# In[54]:


data_2 = bq2df(query)


# In[55]:


data_2


# In[60]:


query = """
  SELECT
      fullVisitorId AS visitor_id,
      visitId AS visit_id,
      date,
      product.v2ProductCategory AS product_category,
      product.v2ProductName AS product_name,
      product.productSKU AS product_sku,
      product.productPrice/1e6 AS product_price,
      product.productQuantity AS product_quantity,
      product.productRevenue/1e6 AS product_revenue,
      totals.totalTransactionRevenue/1e6 AS total_revenue
  FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`
      , UNNEST(hits) AS hits
      , UNNEST(hits.product) AS product
  WHERE
      _TABLE_SUFFIX BETWEEN '20170401' AND '20170430'
      AND geoNetwork.country = 'United States'
      AND productRevenue IS NOT NULL
  ORDER BY
      4 ASC, 10 DESC
"""


# In[61]:


data_3 = bq2df(query)


# In[79]:


data_3.shape


# In[80]:


data_3.head()


# In[ ]:


df = pd.read_csv(('../data/raw/raw.csv'))


# In[ ]:


df.head()


# In[3]:


import platform


# In[4]:


platform.python_version()


# In[1]:


from google.cloud import bigquery

