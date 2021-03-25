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

q = ('SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
    'LIMIT 100')

df = bqd2df(q)
print(df.head(10))

# Save the data
dir_path = os.getcwd()
print(dir_path)
df.to_csv(os.path.join(dir_path, 'data', 'raw','raw.csv'))






# # Perform a query.
# QUERY = (
#     'SELECT * FROM `dummy24571.merch_store.ga_sessions_*` '
#     'LIMIT 100')
# query_job = client.query(QUERY)  # API request
# rows = query_job.result()  # Waits for query to finish

# # for row in rows:
# #     print(row.name)
# print(rows)

# merch_data_ref = client.dataset('merch_store', project = 'dummy24571')
# print(merch_data_ref)

# merch_dataset = client.get_dataset(merch_data_ref)
# print(merch_dataset)