
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import scipy
import pickle
import os
dir_path = os.path.dirname(os.getcwd())
# import sys  
# sys.path.append(os.path.join(dir_path, "src"))
# from utils import create_map


# In[4]:


products = pd.read_csv(os.path.join(dir_path, 'data', 'raw','product.csv'))
products.head()


# In[5]:


norm_df = pd.read_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))
norm_df.head()


# In[6]:


# (data_sparse != sparse_X).nnz==0 
print("------- DATA SUMMARY -------\n")
print("Number of rows in processed data : {}".format(norm_df.shape[0]))
print("Number of columns in processed data : {}\n".format(norm_df.shape[1]))
print("Number of Customers : {}".format(norm_df['visitorId'].nunique()))
print("Total number of products : {}\n".format(norm_df['itemId'].nunique()))


# In[7]:


V = norm_df['visitorId'].nunique() # Get our unique customers
I = norm_df['itemId'].nunique()

visitor_mapper = dict(zip(np.sort(norm_df.visitorId.unique()), list(range(V))))
item_mapper = dict(zip(np.sort(norm_df.itemId.unique()), list(range(I))))

visitor_name = dict(zip(list(range(V)), np.sort(norm_df.visitorId.unique())))
item_name = dict(zip(list(range(I)), np.sort(norm_df.itemId.unique())))

# Get the associated row indices
visitor_index = [visitor_mapper[i] for i in norm_df['visitorId']]
# Get the associated column indices
item_index = [item_mapper[i] for i in norm_df['itemId']]

## Map product ID's with product names
itemId_name = dict(zip(products['itemId'], products['itemName']))
## Inverse of 'itemId_name'
itemName_Id = dict(zip(products['itemName'], products['itemId']))


# In[8]:


## Finds the product name regardless of spelling mistake
def item_finder(itemName):
    all_names = products["itemName"].tolist()
    closest_match = process.extractOne(itemName,all_names)
    return closest_match[0]

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


# Since our goal is to use Cosine Similarity to measure how close Visitor are from each other, we need to transform our dataset from a dense to a sparse representation. In order to achieve that each Visitor needs to be represented by a single row in the dataset so that the columns are the session duration of the Visitor to each different item.

# In[9]:


def create_map(norm_df, item_item = False):
    if item_item == False:
        X = csr_matrix((norm_df["normalized_session_duration"], (visitor_index,item_index)), shape=(V, I))
        return X
    else:
        X = csr_matrix((norm_df["normalized_session_duration"], (item_index, visitor_index)), shape=(I,V))
        return X


# In[10]:


sparse_X =  create_map(norm_df)


# ### Calculating the distance among Visitors

# In[11]:


# calculate similarity between each row that is similar visitors
similarities_sparse = cosine_similarity(sparse_X, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


# Here we will use dense_output=False to have the output as a SciPy sparse matrix, this is a step that we are taking to make sure that our matrix fits in memory, otherwise the output would be a numpy ndarray which isn’t as efficient for storing large and sparse data.
# 
# The shape of our similarities_sparse is (# of visitorID, # of visitorID) and the values are the similarity scores computed for each Visitor against every other Visitor in the dataset.

# In[12]:


similarities_sparse.shape


# Next for every Visitor we need to get the top K most similar Visitor so that we can look at which items they liked and make suggestions - that’s where the actual Collaborative Filtering happens.
# 
# The method top_n_idx_sparse below takes as input a scipy.csr_matrix and returns the top K highest indexes in each row, thats where we get the most similar visitor for each visitor in our Dataset

# In[13]:


###  Return index of top n values in each row of a sparse matrix
def top_n_idx_sparse(matrix, n):
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


# In[14]:


visitor_visitor_similar = top_n_idx_sparse(similarities_sparse, 5)
visitor_visitor_similar


# In[15]:


### Top 5 similar visitors to the visitor
visitor_visitor_similar_dict = {}
for idx, val in enumerate(visitor_visitor_similar):
        visitor_visitor_similar_dict.update({idx: val.tolist()})
visitor_visitor_similar_dict


# In[14]:


## Liked item by a perticular user
def liked_items(visitorId, dataframe = norm_df):
  user_id = (visitor_name[visitorId])
  likes = dataframe[norm_df['visitorId'] == user_id]['itemId'].values.tolist()
  liked = []
  for i in likes:
    liked.append(itemId_name[i])
  return liked
liked_items(16, norm_df)


# In[15]:


def get_top_items(n, df=norm_df):
    most_viewed_itemsId = df.groupby('itemId').count()
#     most_viewed_itemsId.sort_values('visitorId', ascending=False).head(10)
    most_viewed_itemId_sorted = most_viewed_itemsId.sort_values('visitorId',ascending=False)

    top_items_id = []
    top_items_name = []
    for i in range(0,n):
        top_items_id.append(most_viewed_itemId_sorted.index[i])
        product_name = itemId_name[most_viewed_itemId_sorted.index[i]]
        top_items_name.append(product_name)
    return top_items_id, top_items_name # Return the top item Ids and Item names from df 


# In[16]:


top_items_id, top_items_name = get_top_items(10, norm_df)
print("Top 10 most viewed items are :\n {}".format(top_items_name))
print("\n")
print("Top 10 most viewed item Ids are :\n {}".format(top_items_id))


# In[17]:


## Lets find similar items now
item_sparse_X = create_map(norm_df, item_item = True)
item_sparse_X


# In[18]:


item_sparse = cosine_similarity(item_sparse_X, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(item_sparse))


# In[19]:


### Top 5 similar items to the item
item_item_similar = top_n_idx_sparse(item_sparse, 5)
item_item_similar_dict = {}
for idx, val in enumerate(item_item_similar):
        item_item_similar_dict.update({idx: val.tolist()})
item_item_similar_dict


# In[20]:


def recommend_product(q): ## q is Visitor index number
    if q >= V:
        top_items_id, top_items_name = get_top_items(10, norm_df)
        print("Recommend top selling products : \n{}".format(top_items_name))
    else:
        other_likes = []
        q_likes = []
        for j in liked_items(q, norm_df):
            q_likes.append(j)
        print("Items liked by visitor {} are : \n {}".format(q, q_likes))
        for i in visitor_visitor_similar_dict[q]:
            if i != q:
                for p in liked_items(i, norm_df):
                    if p not in other_likes:
                        other_likes.append(p)
        print("\n")        
        print("Items liked by similar visitors \n{}".format(other_likes))
        recom = []
        for item in (other_likes):
            if item not in q_likes:
                recom.append(item)

        if len(recom) == 0:
            for q_like in q_likes:
                item_indx = get_item_index(q_like)
                for ele in item_item_similar_dict[item_indx]:
        #             print(get_itemName(i))
                    if get_itemName(ele) not in q_likes:
                        recom.append(get_itemName(ele))

            print("\n")                
            print("New Product recommendation for you \n{}".format(recom))

        else:
            print("\n")
            print("Product recommendation for you \n{}".format(recom))
            print("\n")
        return recom


# In[21]:


recommendations = recommend_product(585)


# In[22]:


### Check if all the items recommended by the cosine similarily method exist in the top selling items
check =  all(item in top_items_name for item in recommendations)
 
if check is True:
    print("The list {} contains all elements of the list {}".format(top_items_name, recommendations))    
else :
    print("No, top_items_name doesn't have all elements of the recommendations.")


# In[153]:


# ## Save the similarities_sparse matrics
# # save_npz(os.path.join(dir_path, 'data', 'user_user_similarity'), similarities_sparse)

# visitor_visitor_similar = np.load((os.path.join(dir_path, 'data','user_user_similarity.npz')))
# print(user_user_similar)

