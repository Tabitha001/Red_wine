#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[2]:


red_wine_df = pd.read_csv('./Downloads/python/REDwinequality.csv', sep = ';')
red_wine_df


# In[ ]:





# # Getting information about the data

# In[3]:


#names of columns present
red_wine_df.columns


# In[4]:


red_wine_df.shape


# The dataset contains 1599 rows and 12 columns

# In[5]:


red_wine_df.info()


# train_loan_df.info() gives us information about the dataset.
# 
# The dataset contains 11 columns with the float datatype, 1 column with the int (integer datatype)

# In[6]:


red_wine_df.describe()


# train_loan_df.describe() gives the statistical information of the numerical columns (float and int). 

# In[7]:


#number of unique values per column
red_wine_df.nunique()


# In[8]:


red_wine_df['quality'].value_counts()


# # Data cleaning

# In[9]:


#checking for missing values in the dataset
red_wine_df.isna().sum()


# In[10]:


#checking for missing values in the dataset
red_wine_df.isna().sum().sum()


# There are no missing values in the data set

# In[11]:


modified_quality = []
for i in red_wine_df['quality']:
    #good 
    if i >= 7:
        modified_quality.append(1)
    #fair
    elif i >=5 and i <=6:
        modified_quality.append(2)
    #bad
    else:
        modified_quality.append(3)
        
red_wine_df['modified_quality'] = modified_quality


# In[12]:


red_wine_df = red_wine_df.drop('quality', axis = 1)
red_wine_df


# # VISUALIZATIONS OF COLUMNS
# # (drawing insights)

# In[13]:


plt.figure(figsize = (12,10))
sns.countplot(red_wine_df['modified_quality']);


# In[14]:


plt.figure(figsize = (12,10))
fig = px.histogram(red_wine_df, x = 'fixed acidity',
                   marginal = 'box',
                   title = 'FIXED ACIDITY')
fig.update_layout(bargap = 0.3)

fig.show();


# In[15]:


plt.figure(figsize = (12,10))
fig = px.histogram(red_wine_df, x = 'volatile acidity',
                   marginal = 'box',
                   title = 'VOLATILE ACIDITY')
fig.update_layout(bargap = 0.3)

fig.show();


# In[16]:


plt.figure(figsize = (12,10))
fig = px.histogram(red_wine_df, x = 'alcohol',
                   marginal = 'box',
                   title = 'ALCOHOL')
fig.update_layout(bargap = 0.3)

fig.show();


# In[17]:


plt.figure(figsize = (12,10))
fig = px.histogram(red_wine_df, x = 'pH',
                   marginal = 'box',
                   title = 'WINE PH')
fig.update_layout(bargap = 0.3)

fig.show();


# In[18]:


#CHECKING FOR correlation BETWEE COLUMNS
plt.figure(figsize=(12,10))
sns.heatmap(red_wine_df.corr(),cmap = 'Blues', annot=True)
plt.xticks(size=12, color = 'white')
plt.yticks(size=12, color = 'white')
plt.title('CORRELATION BETWEEN COLUMNS', size = 16, color = 'white')
plt.show();


# In[ ]:





# In[19]:


X = red_wine_df.drop(['modified_quality'], axis = 1)
y = red_wine_df['modified_quality']


# In[20]:


#splitting dataset into train and test data
from sklearn.model_selection import train_test_split
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # TRAINING MODELS
# 
# MODELS TO BE TRAINED ARE:
# 
# LOGISTIC REGRESSION
# 
# RANDOM FOREST CLASSIFIER
# 
# K NEAREST NEIGHBORS
# 
# Now, instead of repeating lines of code, while fitting and predicting with the models, setup a dictionary with the models in it, and create a function to fit and score the models.

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#dictionary
Models = {'LR': LinearRegression(), 'KNN': KNeighborsClassifier(),
          'Random_forest': RandomForestClassifier(), 'LOG_REG': LogisticRegression()}

#function
def fit_and_score_model(Models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_score = {}
    for name,model in Models.items():
    #name = key, model = values (of the dictionary created)
        model.fit(X_train, y_train) #fit model to data
        model_score[name] = model.score(X_test, y_test)#model score
    return model_score


# In[22]:


score = fit_and_score_model(Models = Models, X_train = X_train, X_test = X_test,
                            y_train = y_train, y_test = y_test)                            
score


# In[23]:


#Now we can visualize the accuracy of our models.
compare_models = pd.DataFrame(score, index = ['accuracy'])
compare_models


# In[24]:


compare_models.T.plot.bar();
#T denotes Transpose.


# # Hyperparameter Tuning

# In[25]:


#KNN
test_scores = []
train_scores = []
#create a list of values for n_neighbors parameter
neighbors = range(1,21)
knn = KNeighborsClassifier()
#loop through the range of neighbors
for i in neighbors:
    knn.set_params(n_neighbors = i ) #sklearn way of adjusting the parameters of a machine learning model
    knn.fit(X_train, y_train) #fit the tuned model to the training set
    #append the scores to the train_scores and test_scores list above respectively
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
print('maximum test_score = {}'.format(max(test_scores)*100))
print('maximum train_score = {}'.format(max(train_scores)*100))


# THERE IS AN OVERFITTING IN THE KNN MODEL TRAINING WHICH IS NOT GOOD ENOUGH FOR A MODEL

# In[26]:


plt.figure(figsize = (12,8))
plt.plot(neighbors,train_scores)
plt.plot(neighbors,test_scores)
plt.title('KNN scores', color = 'white')
plt.xticks(np.arange(1,21,1), color = 'white')
plt.yticks(color = 'white')
plt.xlabel('n_neighbors', color ='white')
plt.ylabel('scores', color ='white')
plt.legend(['train_score','test_score'])
plt.show();


# In[27]:


#numbers of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
#number of features to consider at every split
max_features =  ['auto', 'sqrt']
#maximum number of levels in tree
max_depth = [2,4]
#minimum number of samples required to split  a node
min_samples_split = [2,5]
#minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
#method of selecting samples for training each tree
bootstrap = [True, False]


# In[28]:


param_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap':bootstrap}


# In[29]:


rf_Model = RandomForestClassifier()


# In[30]:


from sklearn.model_selection import RandomizedSearchCV

rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model,
                                   param_distributions = param_grid, cv = 10,
                                  verbose = 2, n_jobs = 4)


# In[ ]:





# In[31]:


rf_RandomGrid.fit(X_train, y_train)


# In[32]:


print('Train score for the tuned random forest hyperparameters is: {}'
      .format(round(rf_RandomGrid.score(X_train, y_train),2)))
      
print('Test score for the tuned random forest hyperparameters is: {}'
      .format(round(rf_RandomGrid.score(X_test, y_test),2)))


# In[33]:


log_model = LogisticRegression(max_iter = 5000)
log_model.fit(X_train, y_train)


# In[34]:


log_model.score(X_train,y_train)


# In[ ]:





# In[35]:


rf_RandomGrid.predict([[6.4,	0.650, 0.10, 2.0, 0.073, 12.5, 33.2, 0.88756, 3.58, 0.59, 8.3]])


# In[37]:


import pickle


# In[39]:


filename = 'RED WINE QUALITY PREDICTION.sav'
pickle.dump(rf_RandomGrid,open(filename, 'wb'))


# In[40]:


loaded_model = pickle.load(open('RED WINE QUALITY PREDICTION.sav', 'rb'))


# In[ ]:


input_data = []
red_wine_data = np.asarray(input_data)
reshaped_red_wine_data = red_wine_data.reshape(1,-1)

