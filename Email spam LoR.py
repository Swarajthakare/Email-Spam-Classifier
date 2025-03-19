#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# #### Data Collection & Pre-Processing

# In[2]:


raw_mail_data = pd.read_csv(r'B:\Machine Learning Projects Datasets\Email spam\mail_data.csv')


# In[3]:


print(raw_mail_data)


# In[4]:


raw_mail_data


# In[5]:


raw_mail_data.isnull().sum()


# In[6]:


# Replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[7]:


# Printing the first 5 rows of the dataframe
mail_data.head()


# In[8]:


#checking the number of rows and columns in the dataframe
mail_data.shape


# ### Label Encoding

# In[9]:


# label spam mail as 0; ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# * spam = 0
# * ham = 1

# In[10]:


# Seprating the data as texts and label

X = mail_data['Message']


Y = mail_data['Category']


# In[11]:


print(X)


# In[12]:


print(Y)


# ### Splitting the data into training data & test data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)


# In[14]:


print(X.shape)


# In[15]:


print(X_train)


# In[16]:


print(X_train.shape)


# In[17]:


print(X_test)


# In[18]:


print(X_test.shape)


# ### Feature Extraction
# 

# In[19]:


# Transform the text data to feature vectors that can be used as input to the logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# Convert Y_train and Y_test values as integers


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[20]:


print(X_train)


# In[21]:


print(X_train_features)


# ### Training the model
# 
# - Logistic Regression

# In[22]:


model = LogisticRegression()


# In[23]:


# Training the logistic regression model with the training data
model.fit(X_train_features, Y_train)


# ### Evaluating the trained model

# In[24]:


# Prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[25]:


print('Acuuracy on training data :', accuracy_on_training_data)


# In[26]:


# Prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[27]:


print('Accuracy on test data :', accuracy_on_test_data)


# ## Building a predictive System

# In[28]:


input_mail = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]


# Convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)


# Making prediction
prediction = model.predict(input_data_features)
print(prediction)



# In[29]:


if (prediction[0] == 1):
    print("ham mail")
else:
    print("Spam mail")


# In[ ]:




