
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from fuzzywuzzy import process
import implicit
import pickle
import os
dir_path = os.path.dirname(os.getcwd())
import sys  
sys.path.append(os.path.join(dir_path, "src"))
from utils import create_map


# In[2]:


products = pd.read_csv(os.path.join(dir_path, 'data', 'raw', 'product.csv'))


# In[43]:


norm_df = pd.read_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))
norm_df.head()


# In[38]:


sparse_X, user_mapper, item_mapper, userIdx_id, itemIdx_id =  create_map(norm_df)


# ### Check how sparse data is

# In[5]:


sparsity = sparse_X.count_nonzero()/(sparse_X.shape[0]*sparse_X.shape[1])
print(f"Matrix sparsity: {round(sparsity*100,2)}%")


# In[6]:


### Save Sparse matrix for future use
save_npz('../data/user_item_matrix.npz', sparse_X)


# In[7]:


## Finds the product name regardless of spelling mistake
def item_finder(itemName):
    all_names = products["itemName"].tolist()
    closest_match = process.extractOne(itemName,all_names)
    return closest_match[0]

## Map product ID's with product names
itemId_name = dict(zip(products['itemId'], products['itemName']))
## Inverse of 'itemId_name'
itemName_Id = dict(zip(products['itemName'], products['itemId']))

I = norm_df['itemId'].nunique()
## Map item index value to item ID
item_name = dict(zip(list(range(I)), np.unique(norm_df["itemId"])))
## Map item Id to index value
item_mapper = dict(zip(np.unique(norm_df["itemId"]), list(range(I))))


## With Item Id get product name
def get_itemName(item_idx):
    item_id = item_name[item_idx]
    product_name = itemId_name[item_id]
    return product_name

## With item name get item index value
def get_item_index(itemName):
    fuzzy_name = item_finder(itemName)
    item_id = itemName_Id[fuzzy_name]
    item_idx = item_mapper[item_id]
    return item_idx


# In[8]:


item_finder("glass")


# In[9]:


get_item_index('glass')


# In[10]:


get_itemName(196)


# ## Build Model

# In[11]:


model = implicit.als.AlternatingLeastSquares(factors=50, calculate_training_loss=False, use_gpu=False)


# In[12]:


model.fit(sparse_X)


# In[32]:


# rr = [(1, 1.0),
#  (6, 0.79173917),
#  (269, 0.77855545),
#  (14, 0.7783902),
#  (268, 0.75167114),
#  (30, 0.6727726),
#  (13, 0.6721482),
#  (346, 0.66535795),
#  (235, 0.65510774),
#  (231, 0.64452976)]


# In[33]:


i = 1 ### User Clicked on item !
recommendations = model.similar_items(1)
print("Because you liked '{}'\n".format(get_itemName(i)))
for r in recommendations:
    if r[0]!= 1:
        print(get_itemName(r[0]))

