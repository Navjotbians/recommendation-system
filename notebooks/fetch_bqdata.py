
# coding: utf-8

# ## Import libraries

# In[214]:


import pandas as pd
from google.cloud import bigquery
import pickle
import os


# In[197]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# In[215]:


client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')


# In[216]:


def bq2df(sql):
    query = client.query(sql) # API request
    results = query.result()
    return results.to_dataframe()


# ### Raw data

# In[217]:


q = ('SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
    'LIMIT 100')


# In[218]:


complete_data = bq2df(q)


# In[232]:


complete_data.head()


# In[78]:


merch_data = client.get_dataset(merch_data_ref)


# In[147]:


tables = [x.table_id for x in client.list_tables(merch_data)]


# In[148]:


len(tables)


# In[149]:


merch_full = client.get_table(merch_data.table('ga_sessions_20160801'))


# In[188]:


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


# In[189]:


dd = bq2df(q)


# In[190]:


dd.head()


# In[191]:


pre_process = """    
    CREATE OR REPLACE TABLE merch_store.aggregate_web_stats AS (
      WITH
        durations AS (
          --calculate pageview durations
          SELECT
            CONCAT(fullVisitorID,'-', 
                 CAST(visitNumber AS STRING),'-', 
                 CAST(hitNumber AS STRING) ) AS visitorId_session_hit,
            LEAD(time, 1) OVER (
              PARTITION BY CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING))
              ORDER BY
              time ASC ) - time AS pageview_duration
          FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_2016*`,
            UNNEST(hits) AS hit 
        ),

        prodview_durations AS (
          --filter for product detail pages only
          SELECT
            CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING)) AS visitorId,
            productSKU AS itemId,
            IFNULL(dur.pageview_duration,
              1) AS pageview_duration,
          FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_2016*` t,
            UNNEST(hits) AS hits,
            UNNEST(hits.product) AS hits_product
          JOIN
            durations dur
          ON
            CONCAT(fullVisitorID,'-',
                   CAST(visitNumber AS STRING),'-',
                   CAST(hitNumber AS STRING)) = dur.visitorId_session_hit
          WHERE
          #action_type: Product detail views = 2
          eCommerceAction.action_type = "2" 
        ),

        aggregate_web_stats AS(
          --sum pageview durations by visitorId, itemId
          SELECT
            visitorId,
            itemId,
            SUM(pageview_duration) AS session_duration
          FROM
            prodview_durations
          GROUP BY
            visitorId,
            itemId )
        SELECT
          *
        FROM
          aggregate_web_stats
    );
    -- Show table
    SELECT
      *
    FROM
      merch_store.aggregate_web_stats

"""
        


# In[192]:


processed_data = bq2df(qt1)


# In[193]:


processed_data.head()


# In[194]:


processed_data.agg(['count', 'size', 'nunique'])


# In[245]:


(processed_data['itemId'].nunique()).count()


# In[235]:


most_popular = popular_products.sort_values('session_duration', ascending=False)
most_popular.head(10)


# ## Train the matrix factorization model

# In[230]:


get_ipython().run_cell_magic('bigquery', '', "\nCREATE OR REPLACE MODEL merch_store.retail_recommender\nOPTIONS(model_type='matrix_factorization', \n        user_col='visitorId', \n        item_col='itemId',\n        rating_col='session_duration',\n        feedback_type='implicit'\n        )\nAS\nSELECT * FROM merch_store.aggregate_web_stats")

