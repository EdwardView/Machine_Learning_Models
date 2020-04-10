# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:15:31 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)       # quoting =3 is for ignoring inverted commas (" ")

                    # Cleaning the text

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
N = len(df)
ps = PorterStemmer()

for i in range(0, N):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

                # Bag of words model 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()      
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

                # Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))