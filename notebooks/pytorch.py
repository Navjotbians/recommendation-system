
# coding: utf-8

# In[18]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
dir_path = os.path.dirname(os.getcwd())


# In[4]:


products = pd.read_csv(os.path.join(dir_path, 'data', 'raw', 'product.csv'))


# In[5]:


df = pd.read_csv(os.path.join(dir_path, 'data', 'normalized_data.csv'))
df.head()


# # Split train and validation

# In[65]:


# from sklearn.model_selection import train_test_split


# In[66]:


# train, val = train_test_split(df , train_size=0.75,random_state=42)


# In[68]:


# split train and validation before encoding
np.random.seed(3)
msk = np.random.rand(len(df)) < 0.8
train = df[msk].copy()
val = df[~msk].copy()


# In[69]:


train.shape


# In[70]:


val.shape


# Assigning continous index Id's to visitor Id's and item Id's in training set, then mapping these index Id's to the visitor Id's and item id's in Valdation dataset. Also removing the visitor Id; which are not present in the training set

# In[71]:


# Encodes a pandas column with continous ids.
def proc_col(col, train_col=None):
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


# In[72]:


# Encodes rating data with continous user and movie ids. 
# If train is provided, encodes df with the same encoding as train.

def encode_data(df, train=None):
    df = df.copy()
    for col_name in ["visitorId", "itemId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


# In[73]:


# encoding the train and validation data
df_train = encode_data(train)
df_val = encode_data(val, train)


# In[74]:


df_train


# In[75]:


df_val


# ## Model

# In[76]:


visitor_count = len(df_train['visitorId'].unique())
item_count = len(df_train['itemId'].unique())


# In[77]:


visitors =  torch.LongTensor(df_train.visitorId.values)
items = torch.LongTensor(df_train.itemId.values)


# In[78]:


## Every nn.Module subclass implements the operations on input data in the forward method
class MatrixMultiplication(nn.Module):                ## defining Matrix multiplication by subclassing nn.Module
  def __init__(self, visitor_count, item_count, embed_size = 100): 
    super(MatrixMultiplication, self).__init__()     ## Initialize matrix Multiplication using __init__ 
    self.visitor_embed = nn.Embedding(visitor_count, embed_size)
    self.item_embed = nn.Embedding(item_count, embed_size)
    self.visitor_embed.weight.data.uniform_(0, 0.05)
    self.item_embed.weight.data.uniform_(0, 0.05)

  def forward(self, v, i):
    v = self.visitor_embed(v)
    i = self.item_embed(i)
    return (v*i).sum(1)


# In[79]:


def training (model, epochs = 2, lr = 0.001, wd = 0.0, unsqueeze = False):
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  model.train()
  for i in range(epochs):
    visitors = torch.LongTensor(df_train.visitorId.values)
    items = torch.LongTensor(df_train.itemId.values)
    session_duration = torch.FloatTensor(df_train.normalized_session_duration.values)
    # print(session_duration.dtype)
    if unsqueeze:
      session_duration.unsqueeze(1)
    out = model(visitors, items)
    # print(y_hat.dtype)
    loss = F.mse_loss(out, session_duration)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
  testing(model, unsqueeze)


# In[80]:


def testing(model, unsqueeze = False):
  model.eval()
  visitors = torch.LongTensor(df_val.visitorId.values)
  items = torch.LongTensor(df_val.itemId.values)
  session_duration = torch.FloatTensor(df_val.normalized_session_duration.values)
  if unsqueeze :
    session_duration.unsqueeze(1)
  out = model(visitors, items)
  loss = F.mse_loss(out , session_duration)
  print("Test loss ; {}".format(loss.item()))


# In[81]:


model = MatrixMultiplication(visitor_count, item_count)
model


# In[82]:


training(model, epochs=5, lr=0.01)


# In[83]:


a = torch.LongTensor([1])
b = torch.LongTensor([0])


# In[84]:


model.eval()
with torch.no_grad():
  out = model(a, b)
print(out)

