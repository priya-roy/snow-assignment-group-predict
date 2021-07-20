#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:23:43 2021

@author: chattpr7
"""


# Data Gathering & Exploration

import nltk
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords
df = pd.read_csv('incident.csv')
df['assignment_group'].fillna('Service Desk', inplace = True) 
data = []

data = df[['short_description', 'assignment_group']]

corouswords = []
lemna = WordNetLemmatizer()

for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['short_description'][i])
    review = review.lower()
    review = review.split()
    review = [lemna.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    
    corouswords.append(review)

# Creating the bag of words model BOG
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corouswords).toarray()

y = data['assignment_group']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(y)
y = pd.Series(le.transform(y))

# Splitting the data into test train split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60, random_state = 0)


# train the test split
from sklearn.naive_bayes import MultinomialNB 
agroup_pridict_model = MultinomialNB().fit(X_train, y_train)

y_pred = agroup_pridict_model.predict(X_test)

# more analysis
from sklearn.metrics import confusion_matrix
confision_m = confusion_matrix(y_test, y_pred)

# findout accuarcy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

# Pickleing model
import pickle

file = open('cv.pkl', 'wb')
pickle.dump(cv, file)
file.close()

file2 = open('clf.pkl', 'wb')
pickle.dump(agroup_pridict_model, file2)
file2.close()

file3 = open('le.pkl', 'wb')
pickle.dump(le, file3)
file3.close()











