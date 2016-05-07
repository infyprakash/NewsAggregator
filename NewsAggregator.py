# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:32:55 2016

@author: prakashchandraprasad
"""
"""
NewsAgreegator dataset
Classification && Clustering problems
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
df=pd.read_csv('/Users/prakashchandraprasad/Desktop/datasets/NewsAggregatorDataset/newsCorpora.csv',delimiter='\t',header=None)
X_train_raw,X_test_raw,y_train,y_test=train_test_split(df[1],df[4])
vec=TfidfVectorizer()
X_train=vec.fit_transform(X_train_raw)
X_test=vec.transform(X_test_raw)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)
for prediction,msg in zip(predictions,X_test_raw):
    print 'Prediction:%s||News:%s'%(prediction,msg)
