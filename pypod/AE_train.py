# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-03 17:09:05
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-08 00:23:06

from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn import svm
from datetime import datetime
import pandas as pd
import numpy as np
import sys

thres = int(sys.argv[1])
rng = np.random.RandomState(10)
conta = 0.1
epochs = 5

def ele_outliers(num):
    # num = 10
    # fileName1 = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/caida-B-50W-5-{}.csv".format(num)
    # fileName2 = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/caida-B-50W-5-{}.csv".format(num)
    fileName1 = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    fileName2 = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/univ1-50W-{0}-{1}.csv".format(5, num)
    # fileName1 = "data/dec-test.csv"
    # fileName2 = "data/bin-test.csv"
    df = pd.read_csv(fileName1)
    dfb = pd.read_csv(fileName2)
    
    #conver to matrix
    X = dfb.values
    X[X=='0'] = -1
    X[X=='1'] = 1
    yr = df['flowSize']

    # thres = int(sys.argv[1])
    
    
    yc = yr.copy(deep=True)
    yc[yr <= thres] = 0
    yc[yr > thres ] = 1
    print("original mice count: ", sum(yc==0))
    print("original elephant count: ", sum(yc==1))

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=10)
    # split train to ele and mice
    X_train_ele = X_train[y_train == 1]
    X_train_mice = X_train[y_train == 0]

    # use mice to fit the model mice: 1, ele: -1
    # clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    # clf = IsolationForest(max_samples=0.2, n_estimators=300, contamination=conta, random_state=rng)
    # clf.fit(X_train_mice)
    clf_name = 'AutoEncoder'
    clf = AutoEncoder(hidden_neurons=[256, 64, 20, 64, 256], epochs=epochs, contamination=conta, random_state=10, verbose=0)
    clf.fit(X_train_mice)

    y_pred_test = clf.predict(X_test)
    # get outlier scores
    y_pred_scores = clf.decision_function(X_test)

    c_matrix = confusion_matrix(y_test, y_pred_test)
    print(c_matrix)
    print(classification_report(y_test, y_pred_test))
    evaluate_print(clf_name, y_pred_test, y_pred_scores)
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)

    print("thres: ", thres)
    print("conta: ", conta)
    print("epoch: ", epochs)
    for i in range(1, 10):
        print("cycle:", i)
        # mice_outliers(i)
        ele_outliers(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
