
# coding: utf-8

# In[1]:


import pandas as pd
from google.cloud import bigquery
import pickle
import os


# In[96]:


client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')


# In[97]:


def bq2df(sql):
    query = client.query(sql) # API request
    results = query.result()
    return results.to_dataframe()


# In[98]:


q = ('SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
    'LIMIT 100')


# In[99]:


complete_data = bq2df(q)


# In[100]:


complete_data.head()


# In[78]:


merch_data = client.get_dataset(merch_data_ref)


# In[79]:


merch_data


# In[80]:


tables = [x.table_id for x in client.list_tables(merch_data)]


# In[81]:


len(tables)


# In[82]:


merch_full = client.get_table(merch_data.table('ga_sessions_20160801'))


# In[83]:


merch_full


# In[28]:


# [command for command in dir(merch_full) if not command.startswith('_')]


# In[12]:


# merch_full.schema


# In[13]:


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


# In[14]:


data_0 = bq2df(query)


# In[15]:


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


# In[16]:


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


# In[17]:


dd = bq2df(q)


# In[23]:


dd.head()


# In[118]:


qq = """    
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
            v2ProductName AS product_name,
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
            product_name,
            itemId,
            SUM(pageview_duration) AS session_duration
          FROM
            prodview_durations
          GROUP BY
            visitorId,product_name,
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
        


# In[120]:


dd1 = bq2df(qq)


# In[140]:


dd1.head()


# In[138]:


dd1.agg(['count', 'size', 'nunique'])


# In[ ]:


dd1


# In[87]:


qt = """
          SELECT
            CONCAT(fullVisitorID,'-', 
                 CAST(visitNumber AS STRING),'-', 
                 CAST(hitNumber AS STRING) ) AS visitorId_session_hit, ,time,
            LEAD(time, 1) OVER (
              PARTITION BY CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING))
              ORDER BY
              time ASC ) - time AS pageview_duration
          FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_2016*`,
            UNNEST(hits) AS hit 
"""


# In[88]:


dt = bq2df(qt)


# In[89]:


dt.head()


# In[126]:


qt1 = """    
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
        


# In[127]:


dt1 = bq2df(qt1)


# In[133]:


dt1.itemId.shape, dd1.itemId.shape


# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE TEMPORARY FUNCTION CLIP_LESS(x FLOAT64, a FLOAT64) AS (\n  IF (x < a, a, x)\n);\nCREATE TEMPORARY FUNCTION CLIP_GT(x FLOAT64, b FLOAT64) AS (\n  IF (x > b, b, x)\n);\nCREATE TEMPORARY FUNCTION CLIP(x FLOAT64, a FLOAT64, b FLOAT64) AS (\n  CLIP_GT(CLIP_LESS(x, a), b)\n);\n\n\nCREATE OR REPLACE TABLE merch_store.recommendations_data \nAS \nWITH normalized_session_duration AS (\n    SELECT APPROX_QUANTILES(session_duration,100)[OFFSET(50)] AS median_duration\n    FROM merch_store.aggregate_web_stats\n)\nSELECT\n   * EXCEPT(session_duration, median_duration),\n   CLIP(0.3 * session_duration / median_duration, 0, 1.0) AS normalized_session_duration\nFROM\n   merch_store.aggregate_web_stats, normalized_session_duration\n')


# In[ ]:


df.agg(['count', 'size', 'nunique'])


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

