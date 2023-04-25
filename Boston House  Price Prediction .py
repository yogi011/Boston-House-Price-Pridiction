#!/usr/bin/env python
# coding: utf-8

# # Boston Housing Price Prediction 

# In[20]:


# import library
import pandas as pd
import numpy as np


# In[21]:


# To load the inbuilt dataset import sklearn library
from sklearn.datasets import load_boston


# In[22]:


#load boston dataset in to housing_data variable
housing_data=load_boston()


# In[23]:


# print the keys of dataset
print(housing_data.keys())


# In[24]:


#print the Describtion of dataset
print(housing_data.DESCR)


# In[30]:


# Assining the housing data to boston variable
boston=housing_data
boston


# In[31]:


#Dataset is in array form than convert it into a Data Frame
boston=pd.DataFrame(housing_data.data)


# In[32]:


# check the dataset with head function, it will return first 5 rows
boston.head()


# In[35]:


# Dataset is in Data frame formate but columname not assign,we assign the colum name
boston.columns=housing_data.feature_names
boston.head()


# In[36]:


#Price colum is not added in the data set ,now we add target values in price colum
boston["Price"]=housing_data.target
boston.head()


# In[37]:


# now data set is in a Data Frame formate
# now find the information about data type
boston.info()


# In[38]:


# find the statical information about data sets
boston.describe()


# In[39]:


# find the null values 
boston.isnull().sum()


# In[106]:


# Data set has no null values
# visulasite the relation between the colums
# import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
sns.heatmap(boston.corr(),cbar=True,square=True,annot=True,cmap="viridis")


# In[103]:


#Plot the density plot for price 
sns.distplot(boston["Price"])


# In[48]:


# split the data set in to features(Independent) and Target(Dependent) variable
# x is independent variable
# drop the Price colum value from x
x=boston.drop(["Price"],axis=1)
x


# In[50]:


#y= Dependent variable
y=boston.Price
y


# # Linear Regression 

# In[52]:


# importing the Linear Regression model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[53]:


# Divide the data set in to Train and Test set , Trainingset is always greater than the testset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
# test_size=0.25 mean we take test size data is 25% and training size data =75%,random_state 10 means model will take data randomly 


# In[54]:


# Now fit the model 
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[55]:


# Model is fit 
# Predict the value on the based of test size of independent data set
y_pred=lr.predict(x_test)
y_pred


# In[58]:


# check the actual values and model predicted values
data_predicted=pd.DataFrame()
data_predicted["Actual"]=y_test
data_predicted["Predicted Values"]=y_pred
data_predicted


# In[60]:


# Find the error percentage and accuracy of our model 
# import the metrics library for finding accuracy of our model
from  sklearn import metrics


# In[62]:


print("MAE=",metrics.mean_absolute_error(y_test,y_pred))
print("MSE=",metrics.mean_squared_error(y_test,y_pred))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R^2=",metrics.r2_score(y_test,y_pred)*100)
# Model prediction Accuracy is 67%


# In[93]:


# find the Intercept for Linear Model
print("Intercetp=",lr.intercept_)


# In[94]:


# Find the Coefficient of Linear Model
coefficents=pd.DataFrame(lr.coef_,x.columns,columns=["Coefficents"])
coefficents


# In[100]:


# Visualize the Price vs Prediction
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_test,y_pred)
plt.xlabel("Price",c='b',fontsize=15)
plt.ylabel("Prediction Price",c="g",fontsize=15)
plt.title("Price vs Predicted Price",c="r",fontsize=20)
plt.show()


# In[ ]:




