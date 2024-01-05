#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


# In[10]:


titanic=pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\TitanicDataset.csv")


# In[11]:


titanic.head(6)


# In[12]:


titanic.shape


# In[13]:


titanic.describe(include='all')


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[15]:


from sklearn import preprocessing


# In[16]:


titanic.isnull().sum()


# In[17]:


titanic=titanic.drop(columns=['Cabin'], axis=1)


# In[18]:


titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)


# In[19]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# When you print the DataFrame, it will display all rows and columns
print(titanic['Age'])


# In[20]:


titanic.head()


# In[21]:


#Replace all the missing values with the mean value
titanic['Age'].mean()


# In[22]:


titanic['Age'].fillna(titanic['Age'].mean, inplace=True)


# In[23]:


titanic['Age'].dtypes


# In[24]:


print(titanic['Embarked'].mode())


# In[25]:


print(titanic['Embarked'].mode()[0])


# In[26]:


titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace =True)


# In[27]:


titanic.isnull().sum()


# In[28]:


titanic['Survived'].value_counts()


# In[30]:


se.set()


# In[31]:


se.countplot(x='Sex', data=titanic)


# In[32]:


se.countplot(x='Survived', data=titanic)  #count plot for the Survived column


# In[33]:


se.countplot(x='Sex', hue='Survived', data=titanic)


# In[34]:


se.countplot(x='Pclass', data=titanic)


# In[35]:


titanic['Embarked'].value_counts()


# In[36]:


#convert dataset's categorical data into the numerical data
titanic.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[37]:


X=titanic.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y=titanic['Survived']


# In[38]:


titanic.head()


# In[39]:


print(X.head())


# In[40]:


#Spliting the dataset into the training and testing dataset
titanic.head()


# In[41]:



# Training model ,prompt: find what is the problem in the X and Y training?

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[43]:


X.head()


# In[44]:


titanic.describe()


# In[45]:


titanic.info()


# In[47]:


titanic['Age'] = pd.to_numeric(titanic['Age'], errors='coerce')
titanic['Age'] = titanic['Age'].astype('float')


# In[48]:


X_train.head()


# In[49]:


pd.set_option('display.max_colwidth', None)

print(titanic['Age'])


# In[50]:


model=LogisticRegression()
model.fit(X_train, Y_train)


# In[51]:


#Accuracy Score
#SURVIVAL PREDICTION 
X_train_prediction=model.predict(X_train)


# In[52]:


print(X_train_prediction)


# In[53]:


training_data_accuracy=accuracy_score(Y_train, X_train_prediction)


# In[54]:


print("Accurate_score_of_training_data", training_data_accuracy)


# In[55]:


X_test_prediction=model.predict(X_test)
print(X_test_prediction)


# In[56]:


testing_data1_accuracy=accuracy_score(Y_test, X_test_prediction)
print("Accuracy of the test data", testing_data1_accuracy)


# In[ ]:




