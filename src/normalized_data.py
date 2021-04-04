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


Q = """

CREATE TEMPORARY FUNCTION CLIP_LESS(x FLOAT64, a FLOAT64) AS (
  IF (x < a, a, x)
);
CREATE TEMPORARY FUNCTION CLIP_GT(x FLOAT64, b FLOAT64) AS (
  IF (x > b, b, x)
);
CREATE TEMPORARY FUNCTION CLIP(x FLOAT64, a FLOAT64, b FLOAT64) AS (
  CLIP_GT(CLIP_LESS(x, a), b)
);


CREATE OR REPLACE TABLE merch_store.recommendations_data 
AS (
WITH normalized_session_duration AS (
    SELECT APPROX_QUANTILES(session_duration,100)[OFFSET(50)] AS median_duration
    FROM merch_store.aggregate_web_stats
)
SELECT
   * EXCEPT(session_duration, median_duration),
   CLIP(0.3 * session_duration / median_duration, 0, 1.0) AS normalized_session_duration
FROM
   merch_store.aggregate_web_stats, normalized_session_duration);
       -- Show table
    SELECT
      *
    FROM
      merch_store.recommendations_data

"""
 
if __name__ == "__main__":
  ## Create table named 'recommendations_data' that do Median normalization on the processed data stored in 'aggregate_web_stats' table
  df = bqd2df(Q)
  print(df.head(10))



