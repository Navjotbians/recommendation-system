
# coding: utf-8

# # Cosine Similarity based recommendation

# Here I will build a recommendation system using Cosine Similarity for a Collaborative-Filtering approach, in this case I can’t specifically call it Machine Learning because there will be no Gradient Descent or any other type of hyperparameter involved. What I will do is just preprocess our data in a smart way and apply some calculation steps.

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


# Get data folder from here
# https://drive.google.com/drive/folders/1hjyAcf3whkc1SZMvGmTmlktxEqBu55JP?usp=sharing

# In[3]:


# Read product information which I will use later to get the product names
products = pd.read_csv(os.path.join(dir_path, 'data','raw', 'product.csv'))
products.head()


# In[4]:


# Read processed data
norm_df = pd.read_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))
norm_df.head()


# In[5]:


# (data_sparse != sparse_X).nnz==0 
print("------- DATA SUMMARY -------\n")
print("Number of rows in processed data : {}".format(norm_df.shape[0]))
print("Number of columns in processed data : {}\n".format(norm_df.shape[1]))
print("Number of Customers : {}".format(norm_df['visitorId'].nunique()))
print("Total number of products : {}\n".format(norm_df['itemId'].nunique()))


# ## Transforming the data

# We need to transform the norm_df dataframe into a visitor-item matrix where rows represent visitors and columns represent items. The cells of this matrix will be populated with implicit feedback: in this case, the time spent by the visitor on item page (normalized_session_duration).

# In[6]:


# Get our unique customers
V = norm_df['visitorId'].nunique() 
# Get number of unique item
I = norm_df['itemId'].nunique() 

# maps visitor id to visitor index
visitor_mapper = dict(zip(np.sort(norm_df.visitorId.unique()), list(range(V))))
# maps item id to item index
item_mapper = dict(zip(np.sort(norm_df.itemId.unique()), list(range(I))))  

# maps visitor index to visitor id
visitor_name = dict(zip(list(range(V)), np.sort(norm_df.visitorId.unique()))) 
# maps item index to item id
item_name = dict(zip(list(range(I)), np.sort(norm_df.itemId.unique()))) 

# Get the associated row indices
visitor_index = [visitor_mapper[i] for i in norm_df['visitorId']]
# Get the associated column indices
item_index = [item_mapper[i] for i in norm_df['itemId']]

## Map product ID's with product names
itemId_name = dict(zip(products['itemId'], products['itemName']))
## Inverse of 'itemId_name'
itemName_Id = dict(zip(products['itemName'], products['itemId']))


# In[7]:


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

# In[8]:


def create_map(norm_df, item_item = False):
    if item_item == False:
        X = csr_matrix((norm_df["normalized_session_duration"], (visitor_index,item_index)), shape=(V, I))
        return X
    else:
        X = csr_matrix((norm_df["normalized_session_duration"], (item_index, visitor_index)), shape=(I,V))
        return X


# In[9]:


sparse_X =  create_map(norm_df)


# ### Calculating the distance among Visitors

# In[ ]:


# calculate similarity between each row that is similar visitors
similarities_sparse = cosine_similarity(sparse_X, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


# Here we will use dense_output=False to have the output as a SciPy sparse matrix, this is a step that we are taking to make sure that our matrix fits in memory, otherwise the output would be a numpy ndarray which isn’t as efficient for storing large and sparse data.
# 
# The shape of our similarities_sparse is (# of visitorID, # of visitorID) and the values are the similarity scores computed for each Visitor against every other Visitor in the dataset.

# In[ ]:


similarities_sparse.shape


# Next for every Visitor we need to get the top K most similar Visitor so that we can look at which items they liked and make suggestions - that’s where the actual Collaborative Filtering happens.
# 
# The method top_n_idx_sparse below takes as input a scipy.csr_matrix and returns the top K highest indexes in each row, thats where we get the most similar visitor for each visitor in our Dataset

# In[ ]:


###  Return index of top n values in each row of a sparse matrix
def top_n_idx_sparse(matrix, n):
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


# In[ ]:


visitor_visitor_similar = top_n_idx_sparse(similarities_sparse, 5)
visitor_visitor_similar


# In[32]:


### Top 5 similar visitors to the visitor
visitor_visitor_similar_dict = {}
for idx, val in enumerate(visitor_visitor_similar):
        visitor_visitor_similar_dict.update({idx: val.tolist()})
visitor_visitor_similar_dict


# Now we have dictionary that contains the visitor index as a key and a list of similar visitor as a value 

# In[33]:


## Liked item by a visitor
def liked_items(visitorId, dataframe = norm_df):
  user_id = (visitor_name[visitorId])
  likes = dataframe[norm_df['visitorId'] == user_id]['itemId'].values.tolist()
  liked = []
  for i in likes:
    liked.append(itemId_name[i])
  return liked
liked_items(16, norm_df)


# This function will be used in making recommendations for getting the liked items by the visitor

# In[34]:


# Find K top popular items
def top_k_popular_items(n, df=norm_df):
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


# In[35]:


top_items_id, top_items_name = top_k_popular_items(10, norm_df)
print("Top 10 most viewed items are :\n {}".format(top_items_name))
print("\n")
print("Top 10 most viewed item Ids are :\n {}".format(top_items_id))


# In[36]:


## Lets find similar items now so that we can make recommendations based on item similarty as well
item_sparse_X = create_map(norm_df, item_item = True)
item_sparse_X


# In[37]:


item_sparse = cosine_similarity(item_sparse_X, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(item_sparse))


# In[38]:


### Top 5 similar items to the item
item_item_similar = top_n_idx_sparse(item_sparse, 5)
item_item_similar_dict = {}
for idx, val in enumerate(item_item_similar):
        item_item_similar_dict.update({idx: val.tolist()})
item_item_similar_dict


# In[45]:


def recommend_product(q): ## q is Visitor index number
    
    # When recommending to new user
    if q >= V:  
        top_items_id, top_items_name = top_k_popular_items(10, norm_df)
        print("Recommend top selling products : \n{}".format(top_items_name))
        
    # Recommendations for existing user
    else:   
        other_likes = []  # Items liked by the similar visitors 
        q_likes = []   # Already liked items by the visitor for whom we are making recommendations
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
        
        recom = []  # Items liked by the similar visitors excluding q_likes
        for item in (other_likes):
            if item not in q_likes:
                recom.append(item)
        
        # In case similar users have no new item to suggest then we will make suggestions based on the item similarity
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


# ## Results

# In[46]:


# recommendation for visitor 585
recommendations = recommend_product(585)


# In[47]:


### Check if all the items recommended by the cosine similarily method exist in the top selling items
check =  all(item in top_items_name for item in recommendations)
 
if check is True:
    print("The list {} contains all elements of the list {}".format(top_items_name, recommendations))    
else :
    print("No, top_items_name doesn't have all elements of the recommendations.")


# This means our method is performing better than popularity based method

# In[45]:


# ## Save the similarities_sparse matrics
# # save_npz(os.path.join(dir_path, 'data', 'user_user_similarity'), similarities_sparse)

# visitor_visitor_similar = np.load((os.path.join(dir_path, 'data','user_user_similarity.npz')))
# print(user_user_similar)

