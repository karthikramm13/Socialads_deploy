#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


ds = pd.read_csv("Social_ads1.csv")
ds.head()


# In[5]:


x = ds.drop('Purchased', axis = 1).values
y = ds['Purchased'].values


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, random_state = 42)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')


# In[8]:


model.fit(x_train,y_train)


# In[10]:


y_pred = model.predict(x_test)
y_pred


# In[11]:


model.score(x_test,y_test)


# In[14]:


import pickle
pickle.dump(model, open('Social_ads1.pkl','wb'))

# Loading model to compare the results
Social_ads = pickle.load(open('Social_ads1.pkl','rb'))
print(Social_ads.predict([[15624510, 19, 19000]]))


# In[ ]:




