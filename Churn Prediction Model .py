#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier


# In[4]:


df = pd.read_csv("C:/Users/Noor Basher/Downloads/Churn_Modelling.csv")

print(df.head())
print(df.info())


# In[5]:


#Drop Irrelevant Columns



df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)


# In[6]:


#Encode Categorical Variables
# Gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Geography: One-Hot Encoding
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


# In[7]:


df


# In[9]:


#Define Features & Target
X = df.drop('Exited', axis=1)   # Exited = Churn
y = df['Exited']


# In[10]:


#Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[13]:


#Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


#Handle Class Imbalance + Train Model
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)


# In[19]:


#Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# In[ ]:




