#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('jiffs_house_price_my_dataset_v1.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


corr_matrix=data.corr()


# In[6]:


corr_matrix['property_value'].sort_values(ascending=False)


# In[7]:


cost=data['property_value'].values
features=data.drop('property_value',axis=1).values


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,cost,test_size=0.2,random_state=42)


# In[9]:


print(x_train)


# In[10]:


from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
x_train = s_scaler.fit_transform(x_train.astype(float))
x_test = s_scaler.transform(x_test.astype(float))


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[12]:


print(regressor.intercept_)
print(regressor.coef_)


# In[13]:


y_pred = regressor.predict(x_test)


# In[14]:


y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df1 = df.head(10)
df1


# In[15]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))


# In[16]:


import numpy as np
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[17]:


print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
ols = LinearRegression()
ols.fit(x_train, y_train)
ols_yhat= ols.predict(x_test)
ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
ridge_yhat = ridge.predict(x_test)
lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
lasso_yhat = lasso.predict(x_test)
bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
bayesian_yhat = bayesian.predict(x_test)
en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
en_yhat = en.predict(x_test)


# In[19]:


ols.score(x_train,y_train)*100


# In[20]:


ols.score(x_test,y_test)*100


# In[21]:


print(y_train)


# In[22]:


print(y_test)


# In[23]:


print(x_test)


# In[24]:


print("OLSAccuracy:-",ols.score(x_test,y_test)*100)


# In[25]:


print("RidgeAccuracy:-",ridge.score(x_test,y_test)*100)
print("LassoAccuracy:-",lasso.score(x_test,y_test)*100)
print("BayesianAccuracy:-",bayesian.score(x_test,y_test)*100)
print("ENAccuracy:-",en.score(x_test,y_test)*100)


# In[26]:


df2 = pd.DataFrame({'Actual':y_test,'OLSPredicted':ols_yhat,'RidgePredicted':ridge_yhat,'LassoPredicted':lasso_yhat,'BayesianPredicted':bayesian_yhat,'ENPredicted':en_yhat})
df3 = df2.head(50)
df3


# In[27]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(y_test, ols_yhat, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(ols_yhat), max(y_test))
p2 = min(min(ols_yhat), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[28]:


index_position = 25
np_y_test = np.array(y_test)
actual = np_y_test[index_position]/1
pred = round(ols_yhat[index_position],2)
diff = round((np_y_test[index_position]/1)-(ols_yhat[index_position]),2)
perc = round(diff/actual*100,2)
print('Actual is: ' + str(actual))
print('Prediction is: ' + str(pred))
print('Difference is: ' + str(diff))
print('Error Percentage is: ' + str(perc)+'%')


# In[29]:


import joblib


# In[30]:


joblib_file = "joblib_ols_Model.pkl"  
joblib.dump(ols, joblib_file)


# In[31]:


joblib_ols_model = joblib.load(joblib_file)
joblib_ols_model


# In[32]:


score = joblib_ols_model.score(x_test, y_test)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = joblib_ols_model.predict(x_test)  

Ypredict


import pickle
pickle.dump(ols,open('housing.pkl','wb'))




