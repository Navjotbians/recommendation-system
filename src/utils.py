import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from google.cloud import bigquery
import pickle
import os

client = bigquery.Client()
merch_data_ref = client.dataset('merch_store', project = 'dummy24571')

# products = pd.read_csv("../data/raw/product.csv")

def bqd2df(sql):
	query = client.query(sql) # API request
	results = query.result()
	return results.to_dataframe()

def create_map(norm_df):
    V = norm_df['visitorId'].nunique()
    I = norm_df['itemId'].nunique()
    
    user_mapper = dict(zip(np.unique(norm_df["visitorId"]), list(range(V))))
    item_mapper = dict(zip(np.unique(norm_df["itemId"]), list(range(I))))
    
    user_name = dict(zip(list(range(V)), np.unique(norm_df["visitorId"])))
    item_name = dict(zip(list(range(I)), np.unique(norm_df["itemId"])))
    
    user_index = [user_mapper[i] for i in norm_df['visitorId']]
    item_index = [item_mapper[i] for i in norm_df['itemId']]
    X = csr_matrix((norm_df["normalized_session_duration"], (item_index, user_index)), shape=(I, V))
    return X, user_mapper, item_mapper, user_name, item_name


