
# coding: utf-8

# In[2]:


import os
import pandas as pd

DATA_PATH="Test"

def load_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "data.csv")
    return pd.read_csv(csv_path)

raw_data=load_data()


# In[3]:


#info about dataset
raw_data.head()
raw_data.info()

# number of labels,change "name_column"
raw_data["name_column"].value_counts

raw_data.describe()


# In[4]:


# create histogram
import matplotlib.pyplot as plt
raw_data.hist(bins=50,figsize=(15,15))
plt.show()


# In[5]:


# create a test set
# if not enough data (<1k), stratified sampling needed

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(raw_data, test_size=0.2,random_state=42)
raw_data_train=train_set.copy()


# In[6]:


# create scatter matrix
# change attributes for your data set
from pandas.plotting import scatter_matrix
attributes=["name_column"]
scatter_matrix(raw_data_train[attributes], figsize=(12,8))


# In[7]:


# change x and y axis for other data _set

raw_data_train.plot(kind="scatter", x="column_name1",
              y="column_name2",alpha=0.1)


# In[8]:


#calculate correlation matrix
corr_matrix=raw_data_train.corr()
corr_matrix['travel_total_price'].sort_values(ascending=False)


# In[9]:


# data cleaning

#delete NaN raws
raw_data_train=raw_data_train.dropna()
#raw_data_train=raw_data_train.drop("column_name1",axis=1)
raw_data_train.describe()
#raw_data_train=raw_data_train.fillna(median, inplace=True)


# In[31]:


# for one hot encoder

# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# raw_data_cat_1hot=encoder.fit_transform(categories_vector)


# In[ ]:


# feature scaling - min-max scaling (normalization) and standardization

