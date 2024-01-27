#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


df = pd.read_csv("car data.csv")


# In[7]:


df.head()


# ## Data Preprocessing :- 

# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# ## EDA :- 

# In[11]:


df['Owner'].value_counts()


# In[12]:


#Exploring Categorical Features
df['Car_Name'].value_counts()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df['Fuel_Type'].value_counts()


# In[15]:


sns.countplot(x='Fuel_Type', data=df)
plt.show()


# In[16]:


df['Selling_type'].value_counts()


# In[17]:


sns.countplot(x='Selling_type', data=df)
plt.show()


# In[18]:


df['Transmission'].value_counts()


# In[19]:


sns.countplot(x='Transmission', data=df)
plt.show()


# In[20]:


#Exploring Numerical features
df.hist(bins=20, figsize=(15, 10))
plt.show()


# In[21]:


#Finding relationships between different numerical features and our target features
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(x='Year', y='Selling_Price', data=df)
plt.subplot(1, 3, 2)
sns.scatterplot(x='Present_Price', y='Selling_Price', data=df)
plt.subplot(1, 3, 3)
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=df)
plt.tight_layout()
plt.show()


# In[22]:


#Finding Relationship between Cars and it's Selling price using BOXPlot
plt.figure(figsize=(16,12))
sns.boxplot(x='Car_Name', y='Selling_Price', data=df)
plt.xticks(rotation=90)  
plt.show()


# In[23]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df)
plt.title('Relationship between Fuel type and Selling Price')
plt.subplot(1, 2, 2)
sns.boxplot(x='Transmission', y='Selling_Price', data=df)
plt.title('Relationship between Transmission and Selling price')
plt.tight_layout()
plt.show()


# ## Model Building :- 

# In[24]:


#Split the datset into features
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


# In[25]:


# One-hot encoding categorical values into numerical values
X_encoded = pd.get_dummies(X, columns=['Fuel_Type', 'Selling_type', 'Transmission','Car_Name'], prefix=['Fuel', 'Selling', 'Transmission','Cars'])


# In[26]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[27]:


#Train a Regression Model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[28]:


y_pred_linear = linear_model.predict(X_test)


# In[29]:


#Evaluating the Regression Model
from sklearn.metrics import mean_squared_error
from math import sqrt
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = sqrt(mse_linear)
print(f'Linear Regression RMSE: {rmse_linear}')


# In[30]:


#Train a Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


# In[31]:


y_pred_rf = rf_model.predict(X_test)


# In[32]:


#Evaluating the Random Forest Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = sqrt(mse_rf)
print(f'Random Forest RMSE: {rmse_rf}')


# In[33]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price (Random Forest)')
plt.title('Actual vs. Predicted Selling Price (Random Forest)')
plt.show()


# In[ ]:




