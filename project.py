#!/usr/bin/env python
# coding: utf-8

# ## TELCO USERS CLASSIFICATION

# ### IMPORTING LIBRARIES

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# ### IMPORTING DATA SET

# In[5]:


df=pd.read_excel('D:\Major Project\callers.xlsx')


# In[6]:


df


# In[7]:


df.columns


# ### DATA SET DESCRIPTION
# #### The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left or stayed for their service.

# #### Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# 
# #### The raw data contains 7043 rows (customers) and 21 columns (features).
# 
# #### The “Churn” column is our target.

# #### Column and their Description :
# 
# #### Customer ID : A unique ID that identifies each customer.
# 
# #### Gender : Whether the customer is a male or a female
# 
# #### SeniorCitizen : Whether the customer is a senior citizen or not (1, 0)
# 
# #### Partner : Whether the customer has a partner or not (Yes, No)
# 
# #### Dependents : Whether the customer has dependents or not (Yes, No)
# 
# #### Tenure : Number of months the customer has stayed with the company
# 
# #### PhoneService : Whether the customer has a phone service or not (Yes, No)
# 
# #### MultipleLines : Whether the customer has multiple lines or not (Yes, No, No phone service)
# 
# #### InternetService : Customer’s internet service provider (DSL, Fiber optic, No)
# 
# #### OnlineSecurity : Whether the customer has online security or not (Yes, No, No internet service)
# 
# #### OnlineBackup : Whether the customer has online backup or not (Yes, No, No internet service)
# 
# #### DeviceProtection : Whether the customer has device protection or not (Yes, No, No internet service)
# 
# #### TechSupport : Whether the customer has tech support or not (Yes, No, No internet service)
# 
# #### StreamingTV : Whether the customer has streaming TV or not (Yes, No, No internet service)
# 
# #### StreamingMovies : Whether the customer has streaming movies or not (Yes, No, No internet service)
# 
# #### Contract : The contract term of the customer (Month-to-month, One year, Two year)
# 
# #### PaperlessBilling : Whether the customer has paperless billing or not (Yes, No)
# 
# #### PaymentMethod : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# 
# #### MonthlyCharges : The amount charged to the customer monthly
# 
# #### TotalCharges : The total amount charged to the customer
# 
# #### Churn : Whether the customer churned or not (Yes or No)

# ### DATA ANALYSIS

# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


for i in df.columns:
    print(df[i].unique())


# #### Encoding the Categorical Features data to Numerical data

# In[11]:


from sklearn.preprocessing import OneHotEncoder


# In[12]:


Gender={'Male':1,'Female':0}


# In[13]:


Gender


# In[14]:


df.replace(Gender,inplace=True)


# In[15]:


df['gender']


# In[16]:


Partner={'Yes':1,'No':0}


# In[17]:


df.replace(Partner,inplace=True)


# In[18]:


Dependants={'Yes':1,'No':0}


# In[19]:


df.replace(Dependants,inplace=True)


# In[20]:


df['Dependents']


# In[21]:


PhoneService={'Yes':1,'No':0}


# In[22]:


df.replace(PhoneService,inplace=True)


# In[23]:


MultipleLines={'Yes':1,'No':0,'No phone service':2}


# In[24]:


df.replace(MultipleLines,inplace=True)


# In[25]:


InternetService={'No':0, 'Fiber optic':1, 'DSL':2}


# In[26]:


df.replace(InternetService,inplace=True)


# In[27]:


OnlineSecurity={'Yes':1,'No':0,'No internet service':2}
OnlineBackup={'Yes':1,'No':0,'No internet service':2}
DeviceProtection={'Yes':1,'No':0,'No internet service':2}
TechSupport={'Yes':1,'No':0,'No internet service':2}
StreamingTV={'Yes':1,'No':0,'No internet service':2}
StreamingMovies={'Yes':1,'No':0,'No internet service':2}


# In[28]:


df.replace(OnlineSecurity,inplace=True)
df.replace(OnlineBackup,inplace=True)
df.replace(DeviceProtection,inplace=True)
df.replace(TechSupport,inplace=True)
df.replace(StreamingTV,inplace=True)
df.replace(StreamingMovies,inplace=True)


# In[29]:


Contract={'Two year':24, 'Month-to-month':1,'One year':12}
PaperlessBilling={'Yes':1,'No':0}
Churn={'Yes':1,'No':0}


# In[30]:


df.replace(Contract,inplace=True)
df.replace(PaperlessBilling,inplace=True)
df.replace(Churn,inplace=True)


# In[31]:


df.head()


# In[32]:


df.drop(columns=['PaymentMethod'],inplace=True)


# In[33]:


df.head(5)


# In[34]:


df.TotalCharges.dtypes


# In[35]:


df['MonthlyCharges']=df['MonthlyCharges'].astype(int)


# In[36]:


df['MonthlyCharges'].dtypes


# In[37]:


df['MonthlyCharges']


# In[38]:


df.head()


# In[39]:


df.drop(columns=['TotalCharges'],inplace=True)


# In[40]:


df.head(5)


# In[41]:


df.dtypes


# In[42]:


plt.figure(figsize=(17,10))
sns.heatmap(df.corr())


# In[43]:


sns.distplot(df['MultipleLines'],kde=True)


# In[44]:


sns.distplot(df['InternetService'],kde=True,color='black')


# In[45]:


sns.distplot(df['OnlineSecurity'],color='Orange')


# In[46]:


sns.distplot(df['OnlineBackup'])


# In[47]:


sns.distplot(df['DeviceProtection'],color='green')


# In[48]:


sns.distplot(df['TechSupport'],kde=True,color='red')


# In[49]:


sns.distplot(df['StreamingTV'],kde=True,color='violet')


# In[50]:


sns.distplot(df['StreamingMovies'],kde=True,color='purple')


# In[51]:


sns.distplot(df['Contract'],kde=True,color='red')


# In[52]:


sns.distplot(df['Churn'],kde=True)


# In[53]:


print("Number of rows after data cleaning = ",len(df))
print("Number of columns after data cleaning = ",len(df.columns))


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X=df.iloc[:,0:-1].values


# In[56]:


X


# In[57]:


y=df.iloc[:,-1].values


# In[58]:


y


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[60]:


X_train


# In[61]:


X_test


# In[62]:


y_train


# In[63]:


y_test


# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


lr=LogisticRegression()


# In[66]:


tel=lr.fit(X_train,y_train)


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


y_pred=lr.predict(X_test)


# In[69]:


y_pred


# In[70]:


cm=confusion_matrix(y_test,y_pred)


# In[71]:


cm


# In[72]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[73]:


df.head(4)


# In[74]:


lr.predict([[6344,0,1,1,0,38,1,1,1,0,0,0,0,0,0,1,1,74]])


# In[76]:


import pickle


# In[77]:


with open('telco.pkl','wb') as f:
    pickle.dump(lr,f)
 
 
 
lr_model = pickle.load(open('telco.pkl','rb'))
 


# In[ ]:




