#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 21:25:49 2017

@author: saurabh
"""
#
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
style.use('ggplot')
fig = plt.figure()

import numpy as np
df = pd.read_csv("/media/saurabh/04A49C18A49C0E74/DigitRecognizer/train.csv")
test = pd.read_csv("/media/saurabh/04A49C18A49C0E74/DigitRecognizer/test.csv")

columns = df.columns
columns = columns[1:]
X = df[columns]
y = df.label

#===============Random Forest======================
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X,y)
#
#res =rf.predict(test)
#===================================================


#==============KNN==================================
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
res = neigh.predict(test )
#===================================================


sub = pd.read_csv("/media/saurabh/04A49C18A49C0E74/cutiema007203/DigitRecognizer/sample_submission.csv")
sub.Label = res
sub.to_csv("digit.csv",sep=',',index=False)


