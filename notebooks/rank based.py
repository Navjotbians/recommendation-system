
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


# In[3]:


norm_df = pd.read_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))
norm_df.head()


# In[4]:


sparse_X, user_mapper, item_mapper, userIdx_id, itemIdx_id =  create_map(norm_df)


# In[5]:


def get_top_items(n, df=norm_df):
    most_viewed_itemsId = df.groupby('itemId').count()
    most_viewed_itemsId.sort_values('visitorId', ascending=False).head(10)
    most_viewed_itemId_sorted = most_viewed_itemsId.sort_values('visitorId',ascending=False)
    
    ## Map product ID's with product names
    itemId_name = dict(zip(products['itemId'], products['itemName']))
    ## Inverse of 'itemId_name'
    itemName_Id = dict(zip(products['itemName'], products['itemId']))
    
    top_items_id = []
    top_items_name = []
    for i in range(0,n):
        top_items_id.append(most_viewed_itemId_sorted.index[i])
        product_name = itemId_name[most_viewed_itemId_sorted.index[i]]
        top_items_name.append(product_name)

    
    return top_items_id, top_items_name # Return the top item Ids and Item names from df 


# In[6]:


top_items_id, top_items_name = get_top_items(10, norm_df)
print("Top 10 most viewed items are :\n {}".format(top_items_name))
print("\n")
print("Top 10 most viewed item Ids are :\n {}".format(top_items_id))


# ### Check how sparse data is

# In[11]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import scipy


# In[12]:


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


# In[13]:


V = norm_df['visitorId'].nunique()
I = norm_df['itemId'].nunique()
    
user_mapper = dict(zip(np.unique(norm_df["visitorId"]), list(range(V))))
item_mapper = dict(zip(np.unique(norm_df["itemId"]), list(range(I))))
    
user_name = dict(zip(list(range(V)), np.unique(norm_df["visitorId"])))
item_name = dict(zip(list(range(I)), np.unique(norm_df["itemId"])))
    
user_index = [user_mapper[i] for i in norm_df['visitorId']]
item_index = [item_mapper[i] for i in norm_df['itemId']]


# In[40]:


###  Return index of top n values in each row of a sparse matrix
def top_n_idx_sparse(matrix, n):
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


# In[ ]:


item_sparse = cosine_similarity(sparse_X, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(item_sparse))


# In[17]:


sparsity = X.count_nonzero()/(sparse_X.shape[0]*sparse_X.shape[1])
print(f"Matrix sparsity: {round(sparsity*100,2)}%")


# In[ ]:


### Top 5 similar items to the item
item_item_similar = top_n_idx_sparse(item_sparse, 5)
item_item_similar_dict = {}
for idx, val in enumerate(item_item_similar):
        item_item_similar_dict.update({idx: val.tolist()})
item_item_similar_dict


# In[28]:


# Get item names
for i in item_item_similar_dict[1]:
    print(get_itemName(i))


# In[30]:


### Convert into array but menory doesn't allow
# print(sparse_X.toarray())


# In[31]:


# similarities = cosine_similarity(X)
# print('pairwise dense output:\n {}\n'.format(similarities))


# In[ ]:


## Also can output sparse matrices
similarities_sparse = cosine_similarity(X,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


# In[ ]:


## Save the similarities_sparse matrics
# save_npz(os.path.join(dir_path, 'data', 'user_user_similarity'), similarities_sparse)


# In[ ]:


user_user_similar = top_n_idx_sparse(similarities_sparse, 10)
user_user_similar_dict = {}
for idx, val in enumerate(user_user_similar):
        user_user_similar_dict.update({idx: val.tolist()})
user_user_similar_dict


# In[ ]:


# # gets actual user ids from data based on sparse matrix position index
# similar_users_final = {}
# for user, similar_users in user_user_similar_dict.items():
#     idx = user_name[user]
#     values = []
#     for value in similar_users:
#         values.append(user_name[value])

#     similar_users_final.update({idx: values})
# similar_users_final


# In[35]:


## Liked item by a perticular user
def liked_items(visitorId, dataframe = norm_df):
  user_id = (user_name[visitorId])
  likes = dataframe[norm_df['visitorId'] == user_id]['itemId'].values.tolist()
  liked = []
  for i in likes:
    liked.append(itemId_name[i])
  return liked
liked_items(1, norm_df)


# In[36]:


def recommend_product(q): ## q is user index number
    if q >= V:
        top_items_id, top_items_name = get_top_items(10, norm_df)
        print("Recommend top selling products : \n{}".format(top_items_name))
    else:
        other_likes = []
        q_likes = []
        for j in liked_items(q, norm_df):
            q_likes.append(j)
        print("Items liked by user {} are : \n {}".format(q, q_likes))
        for i in user_user_similar_dict[q]:
            if i != q:
                for p in liked_items(i, norm_df):
                    if p not in other_likes:
                        other_likes.append(p)
        print("\n")        
        print("Items liked by similar users \n{}".format(other_likes))
        recom = []
        for item in (other_likes):
            if item not in q_likes:
                recom.append(item)

        if len(recom) == 0:
            for i in q_likes:
                item_indx = get_item_index(i)
                item_item_similar_dict[item_indx]
                for i in item_item_similar_dict[1]:
        #             print(get_itemName(i))
                    if get_itemName(i) not in q_likes:
                        recom.append(get_itemName(i))

            print("\n")                
            print("New Product recommendation for you \n{}".format(recom))

        else:
            print("\n")
            print("Product recommendation for you \n{}".format(recom))
            print("\n")


# In[37]:


# lo = ['Waterpoof Gear Bag', "Google Men's 100% Cotton Short Sleeve Hero Tee White", 'Waterproof Gear Bag']
# item_finder("Waterprof Gear Bag")


# In[39]:


recommend_product(55)

