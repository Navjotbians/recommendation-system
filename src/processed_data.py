import pandas as pd
from google.cloud import bigquery
import pickle
import os

client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')

def bqd2df(sql):
	query = client.query(sql) # API request
	results = query.result()
	return results.to_dataframe()

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
            itemId)
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
        
if __name__ == "__main__":
  df = bqd2df(pre_process)
  print(df.head(10))

# # Save the data
# dir_path = os.getcwd()
# print(dir_path)
# df.to_csv(os.path.join(dir_path, 'data', 'raw','raw.csv'))



