#!/usr/bin/env python
# coding: utf-8

# # HW4
# 
# 
# ## Part I
# 
# Use the **PopDivas_data.csv** on GitHub to build a prediction model.
# 
# 1. Explore the Data
#     - What patterns/relationships do you notice?
# 2. Choose/Build Your Model (*Decision Tree* OR *KNN*)
#     - Why did you choose this type of model?
#     - Which variables are you including in your model?
#     - Choose a model validation technique and explain why you chose it.
#     - Which variables did you standardize and Why?    
# 3. Evaluate Your Model
#     - State how it did (and what evidence/metric do you have to support that?)
# 
#     
# ## Part II
# 
# Use the **YouTubeKidsVideo.csv** on GitHub to build a Naive Bayes Classifier. This dataset looks at the titles/descriptions of YouTube videos that are (1) and are not (0) meant for kids. The variable KidsVideo is 1 if the video is meant for kids, and 0 if it is not. The other variables are 1 if that word (e.g. "toy", "girl"...etc) is in the title/description of the video, and 0 if it is not.
# 
# 1. Explore the Data
#     - What patterns/relationships do you notice?
# 2. Build your model
#     - Which variables are you including in your model?
#     - Choose a model validation technique and explain why you chose it.
# 3. Evaluate Your Model
#     - How did it do? What evidence/metric do you have to support that?

# # Part I

# In[87]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics 
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold # k-fold cv
from sklearn.model_selection import cross_val_score # cross validation metrics
from sklearn.model_selection import cross_val_predict # cross validation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
#from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('precision', '%.7g')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


divas = pd.read_csv('data/PopDivas_data.csv')
print(divas.head())
print(divas.info())
print(divas.isnull().sum())
print(divas.describe())


# In[89]:


print(divas['artist_name'].unique())
print(divas['key'].unique())


# In[90]:


divas['artist_name'] = divas['artist_name'].astype('category')

label_binary = LabelBinarizer()
lb_results = label_binary.fit_transform(divas["artist_name"])
column_names = label_binary.classes_
# lb_results.shape
# column_names

lb_df = pd.DataFrame(lb_results, columns = column_names)
#print(lb_df)

divas = divas.join(lb_df, lsuffix='index', rsuffix='index')

print(divas.head(20))
#combined.drop(combined.columns[9], axis=1, inplace=True)
column_names = divas.columns[15:]


# In[91]:


divas.head()


# In[92]:


(ggplot(divas, aes(x = "energy", y = "danceability"))
        +geom_point())


# 1a - Seems like there is generally a positive relationship between energy and dancability due to the large cluster in the top right corner

# In[93]:


(ggplot(divas, aes(x = "duration_ms", y = "danceability"))
        +geom_point())


# data is too bunched up to view in this view

# In[94]:


divas['duration_s'] = divas['duration_ms'] / 1000

# get rid of interlude songs
divas = divas[divas['duration_s'] > 60]
print(divas.shape)


# In[95]:


#(divas['duration_s'] > 600).sum()
print(divas.loc[divas['duration_s'] > 600])

# got rid of that mix because it would likely throw off the model
divas = divas[divas['duration_s'] < 600]


# In[8]:


(ggplot(divas, aes(x = "duration_s", y = "danceability"))
        +geom_point())


# 1b - converted to seconds so that its easier to interpret. 
# also removed the destinys child medley so that it wouldnt throw off the model

# In[16]:


g = sns.pairplot(divas)


# 1c - at a glance valence also looks like it has a positve relationship, as well as loudness a little bit.

# In[113]:


predictors = ["valence","duration_s","loudness"]

predictors[2:2] = column_names  
    
#print(predictors)
X = divas[predictors]
y = divas["danceability"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

knn = KNeighborsRegressor()

ks = {"n_neighbors": range(1,30)}

# use grid search to find best parameters
grid = GridSearchCV(knn,ks, scoring = "r2", cv = 5)

zscore = StandardScaler()
zscore.fit(X_train)
    
Xs_train = zscore.transform(X_train)
Xs_test = zscore.transform(X_test)

knnmod = grid.fit(X_train, y_train)

knnmod.best_estimator_.get_params()["n_neighbors"]


# In[114]:


poss_k = [1,2,3,4,5,6,7,8,9,10]
acc = {}

for k in poss_k:
    kf = KFold(n_splits = 5)
    knn2 = KNeighborsRegressor(n_neighbors = k)
    
    acc[k] = np.mean(cross_val_score(knn2, X_train, y_train, cv = kf))

print(acc)

chosen_k = max(acc, key=acc.get)
print(chosen_k)

knn_final = KNeighborsRegressor(n_neighbors = chosen_k)
knn_final.fit(X_train,y_train)

knn_final.score(X_test,y_test)


# Write your responses here
# 
# 2 - I chose KNN as the model to use because I believe artists draw a lot of inspiration from other artists and there might be an interesting relationship between these artists since they were kind of around at the same time and I think knn is an interesting model that might be able to capture this relationship. 

# In[109]:


train_pred = knnmod.predict(X_train)
test_pred = knnmod.predict(X_test)

print('training r2 is:', knnmod.score(X_train, y_train)) #training R2
print('testing r2 is:', knnmod.score(X_test, y_test)) #testing R2

print('\ntrain mse is: ', mean_squared_error(y_train,train_pred))
print('test mse is: ', mean_squared_error(y_test,test_pred))


# In[115]:


train_pred = knn_final.predict(X_train)
test_pred = knn_final.predict(X_test)

print('training r2 is:', knn_final.score(X_train, y_train)) #training R2
print('testing r2 is:', knn_final.score(X_test, y_test)) #testing R2

print('\ntrain mse is: ', mean_squared_error(y_train,train_pred))
print('test mse is: ', mean_squared_error(y_test,test_pred))


# Write your responses here
# 
# The model didnt do too terrible according to the r2 in the k-fold model. The initial model seemed very overfitted to the training data which is why i decided to make a k-fold model as well. I standardized all variables because I couldnt figure out how to choose specific ones. I kept getting errors on the zscore fit method.

# # Part II
# 
# Use the YouTubeKidsVideo.csv on GitHub to build a Naive Bayes Classifier. This dataset looks at the titles/descriptions of YouTube videos that are (1) and are not (0) meant for kids. The variable KidsVideo is 1 if the video is meant for kids, and 0 if it is not. The other variables are 1 if that word (e.g. "toy", "girl"...etc) is in the title/description of the video, and 0 if it is not.
# 
#     Explore the Data
#         What patterns/relationships do you notice?
#     Build your model
#         Which variables are you including in your model?
#         Choose a model validation technique and explain why you chose it.
#     Evaluate Your Model
#         How did it do? What evidence/metric do you have to support that?
# 

# In[31]:


vids = pd.read_csv('data/YouTubeKidsVideo.csv')
print(vids.head())
print(vids.info())
print(vids.isnull().sum())
print(vids.describe())


# In[32]:


kids = vids[vids['kidsVideo'] == 1]
not_kids = vids[vids['kidsVideo'] == 0]

print(kids.shape)
print(not_kids.shape)


# In[33]:


Data = {'total': [kids['cat'].sum(),kids['toy'].sum(),kids['sad'].sum(),kids['girl'].sum(),kids['is'].sum()]}
df = pd.DataFrame(Data,columns=['total'])
my_labels = 'cat','toy','sad', 'girl', 'is'
plt.pie(df,labels=my_labels,autopct='%1.1f%%')
plt.title('Word Prevalance in Title for YouTube Videos MEANT for Kids')
plt.axis('equal')
plt.show()


# In[34]:


Data = {'total': [not_kids['cat'].sum(),not_kids['toy'].sum(),not_kids['sad'].sum(),not_kids['girl'].sum(),not_kids['is'].sum()]}
df = pd.DataFrame(Data,columns=['total'])
my_labels = 'cat','toy','sad', 'girl', 'is'
plt.pie(df,labels=my_labels,autopct='%1.1f%%')
plt.title('Word Prevalance in Title for YouTube Videos NOT MEANT for Kids')
plt.axis('equal')
plt.show()


# data shows that there are more titles with the word toy in the title in videos meant for kids. The opposite is true in the case for videos not meant for children, rather sad is a more common word in titles not meant for kids. 

# In[118]:


predictors = ['toy', 'sad']

X = vids[predictors]
y = vids["kidsVideo"]


kf = KFold(n_splits = 4)
nb = GaussianNB()
acc = []
predictedVals = [] 
for train, test in kf.split(X,y):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y[train]
    y_test = y[test]
    
    
    
    nb.fit(X_train,y_train)
    acc.append(nb.score(X_test,y_test))
    cnf_matrix = confusion_matrix(y_test, y_test)
    print(cnf_matrix)
    


# Write your responses here
# 
# i used the two most prevelant words in the model because they were the most distinct keywords in titles. I chose accuracy as the metric to use in assessing the model because it is a very common metric to determines the overall predicted accuracy of the model.

# In[117]:


print(acc)
print(np.mean(acc))


# Write your responses here
# 
# The model doesnt do too well given a score close to .50 indicating that the model is accurate and useful in half of the time.   

# In[ ]:


get_ipython().system("jupyter nbconvert --output-dir='output/' --to pdf HW4.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to markdown HW4.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to html HW4.ipynb")
get_ipython().system("jupyter nbconvert --output-dir='output/' --to python HW4.ipynb")

