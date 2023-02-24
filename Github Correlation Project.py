#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
plt.style.use ('ggplot')
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (8, 4)

Ad_data = pd.read_csv('Advertising.csv', index_col=0)


# In[2]:


Ad_data.head()


# In[3]:


Ad_data.shape


# In[4]:


Ad_data.describe


# In[5]:


Ad_data.dtypes


# In[6]:


#Checking for Missing Values
for col in Ad_data.columns:
  missing = np.mean(Ad_data[col].isnull())
print('{} - {}%'.format(col, missing))


# In[7]:


#Rename Dataset columns
Ad_data.columns = ['TV','Radio','Newspaper','Sales']
Ad_data.head()


# In[9]:


# Creating a scatter plots with Ads vs Sales

plt.scatter(x=Ad_data['TV'], y=Ad_data['Sales'])
plt.title('TV Ads vs Sales')
plt.xlabel('TV Ad Subscription')
plt.ylabel(' Total Sales')
plt.show()


# In[8]:


plt.scatter(x=Ad_data['Radio'], y=Ad_data['Sales'])
plt.title('Radio Ads vs Sales')
plt.xlabel('Radio Ads subscription')
plt.ylabel('Total Sales')
plt.show()


# In[10]:


plt.scatter(x=Ad_data['Newspaper'], y=Ad_data['Sales'])
plt.title('Newspaper Ads vs Sales')
plt.xlabel('Newspaper Ads subscription')
plt.ylabel('Total sales')
plt.show()


# In[11]:


#ploting the Parameters side by side for a better comparative 

fig,axs = plt.subplots(1,3,sharey=True)
Ad_data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
Ad_data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
Ad_data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


# In[12]:


sns.regplot(x='TV', y='Sales', data=Ad_data)


# In[13]:


sns.regplot(x='Radio', y='Sales', data=Ad_data)


# In[14]:


sns.regplot(x='Newspaper', y='Sales', data=Ad_data)


# In[15]:


#looking at the correlation between Ads spending and Sales
Ad_data.corr(method='pearson')


# In[19]:


correlation_matrix=Ad_data.corr(method='pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title( 'Ads vs Sales Correlation Matric')
plt.xlabel('Ads subscription')
plt.ylabel('Ads subscription')
plt.show()


# In[17]:


#applying linear regression to get the relationship between Sales and Ads spending
feature_cols=['TV']
x= Ad_data[feature_cols]
y=Ad_data.Sales


# In[18]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression() #initializing the Model
lm.fit(x,y) #Fitting the Model on X and Y axis


# In[19]:


#showing that a unit increase in Television Ad spending is associated with a 0.047537 unit increase in Sales.
print(lm.intercept_)
print(lm.coef_)


# In[21]:


#this results shows that a unit increase in Ad Spending is Associated with a 0.0475 Increase in Sales.
#that means getting a predictor for a  #50,000 Increase in Ad spend will Result to a increase in Sales
7.032594 + 0.047537 * 50 

