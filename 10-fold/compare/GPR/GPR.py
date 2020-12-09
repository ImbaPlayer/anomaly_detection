# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-30 17:33:42
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-30 17:40:36
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import pandas as pd
import numpy as np
import sys

clf_type = "DT"

def main():
    fileName1 = "../../autoencoder/data/dec-test.csv"
    fileName2 = "../../autoencoder/data/bin-test.csv"
    df = pd.read_csv(fileName1)
    dfb = pd.read_csv(fileName2)
    
    print(dfb.shape)
    #conver to matrix
    X = dfb.values
    X[X=='0'] = -1
    X[X=='1'] = 1
    yr = df['flowSize']

    # thres = int(sys.argv[1])
    thres = 200
    
    
    yc = yr.copy(deep=True)
    yc[yr <= thres] = -1
    yc[yr > thres ] = 1
    print("original mice count: ", sum(yc==0))
    print("original elephant count: ", sum(yc==1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=10)
    print("train", X_train.shape)
    print("test", X_test.shape)
    if clf_type == "GPR":
        gpr = GaussianProcessRegressor(n_restarts_optimizer=10, random_state=10)
        gpr.fit(X_train, y_train)
        # return the mean
        y_predict = gpr.predict(X_test)
        y_predict[y_predict > 0] = 1
        y_predict[y_predict < 0] = -1
        y_predict_std = gpr.predict(X_test, return_std=True)
        y_predict_cov = gpr.predict(X_test, return_cov=True)
        print(y_predict_std)
        print()
        print(y_predict_cov)
        print()
    elif clf_type == "Naive Bayes":
        clf = GaussianNB()
    elif clf_type == "SVM":
        kernel_type = ['linear', 'poly', 'rbf']
        clf = svm.SVC(kernel='rbf', gamma='scale')
    elif clf_type == "DT":
        tree_criterions = ['gini','entropy']
        clf = DecisionTreeClassifier(criterion = "gini", random_state=10)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(y_predict)
    

if __name__ == "__main__":
    main()