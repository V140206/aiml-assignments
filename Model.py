#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Fake_Bill.csv')
data


# In[3]:


missing_values=data.isna().sum()
missing_values


# In[4]:


duplicate_values=data.duplicated().sum()
duplicate_values


# In[5]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
data['height_diff'] = abs(data['height_left'] - data['height_right'])
data['margin_diff'] = abs(data['margin_up'] - data['margin_low'])
data['diagonal_length_ratio'] = data['diagonal'] / data['length']
print(data[['height_diff', 'margin_diff', 'diagonal_length_ratio']].head())


# In[6]:


scaler = StandardScaler()
columns_to_scale = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length', 'height_diff', 'margin_diff', 'diagonal_length_ratio']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
print(data.head())


# In[7]:


print("Descriptive Statistics:")
print(data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']].describe())


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
plt.figure(figsize=(12, 8))
data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data[feature])
    plt.title(f'Box plot of {feature}')
plt.tight_layout()
plt.show()


# In[9]:


sns.pairplot(data, hue='is_genuine', vars=['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length'])
plt.suptitle('Pair Plot - Relationships Between Features and Authenticity', y=1.02)
plt.show()


# In[10]:


corr_matrix = data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[11]:


sns.countplot(x='is_genuine', data=data)
plt.title('Class Distribution of Genuine vs Fake Bills')
plt.show()


# In[12]:


class_distribution = data['is_genuine'].value_counts(normalize=True)
print("Class Distribution:")
print(class_distribution)
if class_distribution[0] < 0.8:
    X = data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length', 'height_diff', 'margin_diff', 'diagonal_length_ratio']]
    y = data['is_genuine']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("New class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts(normalize=True))


# In[ ]:




