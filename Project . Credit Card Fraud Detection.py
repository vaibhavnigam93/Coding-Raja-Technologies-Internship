#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv(r"E:\Vaibhav\Internship\Coding Raja\creditcard.csv")


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


# dataset informations
credit_card_data.info()


# In[6]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[7]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# This Dataset is highly unblanced
# 
# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[8]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


# statistical measures of the data
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Under-Sampling
# 
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# 
# Number of Fraudulent Transactions --> 492

# In[13]:


legit_sample = legit.sample(n=492, random_state=2)


# Concatenating two DataFrames

# In[14]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# Splitting the data into Features & Targets

# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# Split the data into Training data & Testing Data

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[22]:


model = LogisticRegression(max_iter=1000)


# In[23]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[24]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


# Model Evaluation

# Accuracy Score

# In[25]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[26]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[27]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[28]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




