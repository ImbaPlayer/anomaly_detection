# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-11 19:29:15
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-11 20:47:29
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import pandas as pd
import numpy as np
import sys
from datetime import datetime
# from tqdm import tqdm

def main(num):
    fileName1 = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/caida-A-50W-5-{}.csv".format(num)
    fileName2 = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/caida-A-50W-5-{}.csv".format(num)
    df = pd.read_csv(fileName1)
    dfb = pd.read_csv(fileName2)
    
    #conver to matrix
    X = dfb.values
    X[X=='0'] = -1
    X[X=='1'] = 1
    yr = df['flowSize']

    thres = int(sys.argv[1])
    print("thres: ", thres)
    # thres = 250
    yc = yr.copy(deep=True)
    yc[yr > thres] = 1
    yc[yr <= thres ] = -1
    print("original mice count: ", sum(yc==-1))
    print("original elephant count: ", sum(yc==1))

    #oversampling by smote
    #test train split
    X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=10)
    #oversampling minority class
    #while(sum(y_train==-1) / sum(y_train==1) > 2):
         #mask = (y_train == 1)
         #X_train = np.concatenate((X_train, X_train[mask]), axis=0)
         #y_train = np.concatenate((y_train, y_train[mask]), axis=0)
    # smote = RandomUnderSampler(random_state=10)
    # X_train_sample, y_train_sample = smote.fit_sample(X_train, y_train)
    X_train_sample, y_train_sample = X_train, y_train
    print(sum(y_train==1), sum(y_train==-1), sum(y_test==1), sum(y_test==-1))
    print("sampling:", sum(y_train_sample==1), sum(y_train_sample==-1))
    #neural network
    print("neural network:")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 40), activation='tanh', max_iter=400, random_state=10)
    mlp.fit(X_train_sample, y_train_sample)
    predictions = mlp.predict(X_test)
    c_matrix = confusion_matrix(y_test, predictions)
    print(c_matrix)
    print(classification_report(y_test,predictions))

    #random forest
    print("random forest:")
    rf = RandomForestClassifier(n_estimators=30, class_weight={1:1,-1:1}, random_state=10)
    rf = rf.fit(X_train_sample, y_train_sample)
    predictions = rf.predict(X_test)
    c_matrix = confusion_matrix(y_test, predictions)
    print(c_matrix)
    print(classification_report(y_test,predictions))
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)

    for i in range(10):
        print("cycle:", i)
        # mice_outliers(i)
        main(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
