#!/usr/bin/env python
# coding: utf-8

# ### Importing relavant Libraries

# In[1]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[4]:


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


# ### Data Inspection

# In[5]:


train.shape,test.shape


# ##### As said above we have 8523 rows and 12 columns in Train set whereas Test set has 5681 rows and 11 columns.

# In[6]:


test.apply(lambda x: sum(x.isnull()))


# In[7]:


test.isnull().sum()/test.shape[0] *100


# ##### We have 17% and 28% of missing values in Item weight and Outlet_Size columns respectively

# In[8]:


train.info()


# In[9]:


categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])


# In[10]:


categorical = test.select_dtypes(include =[np.object])
print("Categorical Features in Test Set:",categorical.shape[1])

numerical= test.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Test Set:",numerical.shape[1])


# In[11]:


train.describe()


# In[12]:


test.describe()


# ### Data Cleaning
# #### 1. Item Size

# In[13]:


train.columns


# In[14]:


train['Item_Weight'].isnull().sum(),test['Item_Weight'].isnull().sum()


# In[15]:


plt.figure(figsize=(8,5))
sns.boxplot('Item_Weight', data=train)


# In[16]:


plt.figure(figsize=(8,5))
sns.boxplot('Item_Weight', data=test)


# ##### The Box Plots above clearly show no "Outliers" and hence we can impute the missing values with "Mean"

# In[17]:


train['Item_Weight']= train['Item_Weight'].fillna(train['Item_Weight'].mean())
test['Item_Weight']= test['Item_Weight'].fillna(test['Item_Weight'].mean())


# In[18]:


train['Item_Weight'].isnull().sum(),test['Item_Weight'].isnull().sum()


# ##### We have succesfully imputed the missing values from the column Item_Weight.
# #### 2. Outlet_Size

# In[19]:


train['Outlet_Size'].isnull().sum(),test['Outlet_Size'].isnull().sum()


# In[20]:


print(train['Outlet_Size'].value_counts())
print('******************************************')
print(test['Outlet_Size'].value_counts())


# ##### Since the outlet_size is a categorical column, we can impute the missing values by "Mode"(Most Repeated Value) from the column.

# In[21]:


train['Outlet_Size']= train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
test['Outlet_Size']= test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])


# In[22]:


train['Outlet_Size'].isnull().sum(),test['Outlet_Size'].isnull().sum()


# ##### We have succesfully imputed the missing values from the column Outlet_Size.

# ### Exploratory Data Analysis

# In[23]:


train.head()


# In[24]:


train['Item_Fat_Content'].value_counts()


# ##### We see there are some irregularities in the column and it is needed to fix them

# In[25]:


train['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)


# In[26]:


train['Item_Fat_Content']= train['Item_Fat_Content'].astype(str)


# In[27]:


train['Years_Established'] = train['Outlet_Establishment_Year'].apply(lambda x: 2020 - x) 
test['Years_Established'] = test['Outlet_Establishment_Year'].apply(lambda x: 2020 - x)


# In[28]:


train.head()


# ### Univariate Analysis
# #### 1. Item fat content

# In[29]:


plt.figure(figsize=(8,5))
sns.countplot('Item_Fat_Content',data=train,palette='ocean')


# ### Observations:
#    1. Low fat items are bought more than regular

# #### 2. Item Type

# In[30]:


plt.figure(figsize=(25,7))
sns.countplot('Item_Type',data=train,palette='spring')


# ### Observations:
#   1. Fruits and vegetables are largely sold as people tend to use them on a daily basis
#   2. Snack food too have a good sale.

# #### 3. Outlet Size

# In[31]:


plt.figure(figsize=(8,5))
sns.countplot('Outlet_Size',data=train,palette='summer')


# ### Observations:
#    1. Te Outlets are more of Medium size

# #### 4. Outlet location type

# In[32]:


plt.figure(figsize=(8,5))
sns.countplot('Outlet_Location_Type',data=train,palette='autumn')


# ### Observations:
#     1. Outlets are maximum in number in Tier 3 cities

# #### 5. Outlet Type

# In[33]:


plt.figure(figsize=(8,5))
sns.countplot('Outlet_Type',data=train,palette='autumn')


# ### Observations:
#     1. The outlets are more of Supermarket Type 1

# ### Bivariate Analysis

# In[34]:


train.columns


# #### 1. Item fat

# In[35]:


plt.figure(figsize=(8,5))
sns.barplot('Item_Fat_Content', 'Item_Outlet_Sales', data=train, palette='mako')


# ### Observations
#     1. Both low and regular fat conten items have high sales
# #### 2. Item Visibility

# In[36]:


plt.figure(figsize=(8,5))
plt.scatter('Item_Visibility','Item_Outlet_Sales', data=train)


# ### Observations
#     1. Item visibility has a minimum value of 0.This makes n practical sense coz when a product is being sold in a store, its visibility cannot be 0.
# ##### Let us consider as a missing value and impute it by mean visibility value of that item.

# In[37]:


train['Item_Visibility']= train['Item_Visibility'].replace(0,train['Item_Visibility'].mean())
test['Item_Visibility']= test['Item_Visibility'].replace(0,test['Item_Visibility'].mean())


# In[38]:


plt.figure(figsize=(8,5))
plt.scatter(y='Item_Visibility',x='Item_Outlet_Sales',data=train)
plt.xlabel('Item Outlet Sales')
plt.ylabel('Item Visibility')


# ##### We can see that now visibility is not exactly zero and it has some value indicating that Item is rarely purchased by the customers.
# ##### 3. Item Type

# In[40]:


plt.figure(figsize=(10,12))
sns.barplot(y='Item_Type', x='Item_Outlet_Sales', data=train, palette='flag')


# ##### The products available were Fruits-Veggies and Snack Foods but the sales of Seafood and Starchy Foods seems higher and hence the sales can be improved with having stock of products that are most bought by customers.

# In[41]:


plt.figure(figsize=(8,5))
plt.scatter(y='Item_Outlet_Sales',x='Item_MRP',data=train)
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')


# ### Observation
#     1. Items MRP ranging from 200-250 dollars is having high Sales.
# ##### 4. Outlet Size

# In[42]:


plt.figure(figsize=(8,5))
sns.barplot(x='Outlet_Size',y='Item_Outlet_Sales',data=train,palette='winter')


# ### Observations:
#     1. Sales is greater for medium and high outlet size
# ##### 5.Outlet Location Type

# In[44]:


plt.figure(figsize=(8,5))
sns.barplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=train,palette='plasma')


# ### Obseravtions:
#     1. The Outlet Sales tend to be high for Tier3 and Tier 2 location types but we have only Tier3 locations maximum Outlets.
# ##### 6. Years established

# In[45]:


plt.figure(figsize=(8,5))
sns.barplot(x='Years_Established',y='Item_Outlet_Sales',data=train,palette='viridis')


# ### Observations:
#     1. It is quiet evident that Outlets established 35 years before is having good Sales margin.
#     2.We also have a outlet which was established before 22 years has the lowest sales margin, so established years wouldn't improve the Sales unless the products are sold according to customer's interest.

# ### Multivariate Analysis

# In[46]:


plt.figure(figsize=(25,5))
sns.barplot('Item_Type','Item_Outlet_Sales',hue='Item_Fat_Content',data=train,palette='mako')
plt.legend()


# In[47]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Location_Type','Item_Outlet_Sales',hue='Outlet_Type',data=train,palette='magma')
plt.legend()


# ### Observations:
#     1. The Tier-3 location type has all types of Outlet type and has high sales margin.

# ## Feature Engineering

# In[48]:


train.head()


# In[56]:


le = LabelEncoder()
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']

for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
for i in var_mod:
    test[i] = le.fit_transform(test[i])


# In[57]:


train.head()


# ##### There are some columns that needs to be dropped as they don't seem helping our analysis.

# In[51]:


train = train.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)
test= test.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)


# In[52]:


train.columns


# In[58]:


X= train[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Years_Established']]
y= train['Item_Outlet_Sales']


# In[59]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=22)


# #### Feature Scaling

# In[60]:


features= ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Years_Established']


# ### Linear Regression
# #### Preparing the model and importing necessary packages

# In[62]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# #### Fitting the model

# In[64]:


reg.fit(X_train,y_train)


# ##### Finding accuracy of Linear regression model

# In[65]:


reg.score(X_test,y_test)


# ### Gradient Boosting Regressor
# #### Preparing the model and importing necessary packages

# In[66]:


from sklearn.ensemble import GradientBoostingRegressor
grad= GradientBoostingRegressor(n_estimators=100)


# ##### Fitting the model

# In[67]:


grad.fit(X_train,y_train)


# ##### Finding the accuracy of Gradient Boosting Regressor

# In[68]:


grad.score(X_test,y_test)


# ### Random Forest Regressor
# ##### Preparing the model and importing necessary pacakges

# In[69]:


from sklearn.ensemble import RandomForestRegressor
ran=RandomForestRegressor(n_estimators=50)


# ##### Fitting the model

# In[70]:


ran.fit(X_train,y_train)


# ##### Finding accuracy of Random Forest Model

# In[71]:


ran.score(X_test,y_test)


# ### Conclusion
#  ##### We are given a Big_Mart dataset The aim is to build a predictive model and find out the sales of each product at a particular store. Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
#  ##### First we explore the data performing EDA using various data vizualization tools and draw the necessary conclusions from univariate,bivariate and multivariate analysis
#  ##### After EDA, we perform feature engineering and feature scaling. Intead of using one-hot encoder, we instead use label encoder as the categorical data has been handeled and using one-hot encoder wont make much difference.We then built three models over our datasets and find which one performs the best.
#  ##### Linear regression accuray score: 0.4946245671867815
#  ##### GradientBoostingRegressor accuracy Score: 0.5713935192940436
#  ##### RandomForestRegressor accuracy score: 0.5216971567098128
#  ##### From the above results we conclude that GradientBoostingRegressor has performed the best, thus boosting algorithms efficient for most of the predictive cases.

# In[ ]:



