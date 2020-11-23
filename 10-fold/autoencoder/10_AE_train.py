# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-03 17:09:05
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-16 19:37:52

from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.model_selection import KFold
from datetime import datetime
import pandas as pd
import numpy as np
import sys

# thres = int(sys.argv[1])
elePercent = float(sys.argv[1])
rng = np.random.RandomState(10)
conta = 0.1
epochs = 50
PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]
ALL_TRAIN_TYPE = ["5-tuple", "time", "size", "stat"]

def get_thres(flowSize, elePercent):
    # param flowSize is DataFrame
    np_flowSize = np.array(flowSize)
    quantile = 1 - elePercent
    thres = np.quantile(np_flowSize, quantile)
    return thres
    
def get_col_names(trainType):
    col_names = []
    if trainType == "size":
        statistic_names = ["pkt-size-{}".format(i) for i in range(PACKET_NUMBER)]
        for col_name in statistic_names:
            temp_cols = [col_name + '-{}'.format(i) for i in range(32)]
            col_names.extend(temp_cols)
    elif trainType == "stat":
        statistic_names = ["mean", "var", "min", "max"]
        for col_name in statistic_names:
            temp_cols = [col_name + '-{}'.format(i) for i in range(32)]
            col_names.extend(temp_cols)
    elif trainType == "5-tuple":
        col_names = ["srcIP", "srcPort", "dstIP", "dstPort", "protocol"]
    return col_names
def load_data(dataSetType, trainType, num):
    num = 0
    if trainType == "time":
        # user time interval as features
        fileName1 = "/data/sym/anomaly_detection/data/10-fold/{}/packet-level{}-{}.csv".format(dataSetType, dataSetType, num)
        dfb = pd.read_csv(fileName1)
        yr = dfb['flowSize']
        
        # drop flowSize
        dfb = dfb.drop(["flowSize"], axis=1)
        # convert to matrix
        X = dfb.values()
        
    else:
        fileName1 = "data/dec-test.csv"
        fileName2 = "data/bin-test.csv"
        df = pd.read_csv(fileName1)
        dfb = pd.read_csv(fileName2)

        # get specific cols
        dfb = dfb.loc[:, get_col_names(trainType)]
        
        #conver to matrix
        X = dfb.values
        X[X=='0'] = -1
        X[X=='1'] = 1
        yr = df['flowSize']

        # thres = int(sys.argv[1])
        
    yc = yr.copy(deep=True)
    thres = get_thres(yr, elePercent)
    yc[yr <= thres] = 0
    yc[yr > thres ] = 1
    print("original mice count: ", sum(yc==0))
    print("original elephant count: ", sum(yc==1))
    return X, yc
def ele_outliers(num):
    dataSetType = ALL_DATA_TYPE[0]
    trainType = ALL_TRAIN_TYPE[1]
    
    X, yc = load_data(dataSetType, trainType, num)

    # 10 fold validation
    KF = KFold(n_splits=10, shuffle=True, random_state=10)
    report_list = []
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = yc[train_index], yc[test_index]

        # split into train and test
        # X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=10)
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
        temp_report = classification_report(y_test, y_pred_test, output_dict=True)
        report_list.append(temp_report)
        print(classification_report(y_test, y_pred_test, output_dict=False))
        # evaluate_print(clf_name, y_pred_test, y_pred_scores)
    final_report = get_avg_report(report_list)
    print("final report", final_report)
def get_avg_report(report_list):
    report_array = np.array(report_list)
    np.save('a.npy', report_array)
    
    report_list_0 = []
    report_list_1 = []
    acc_list = []
    result = {}
    for data in report_list:
        report_list_0.append(data['0'])
        report_list_1.append(data['1'])
        acc_list.append(data['accuracy'])
    df_0 = pd.DataFrame(report_list_0)
    df_1 = pd.DataFrame(report_list_1)
    result['0'] = dict(df_0.mean())
    result['1'] = dict(df_1.mean())
    result["accuracy"] = np.mean(acc_list)
    np.save("b.npy", result)
    return result      
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)

    print("thres: ", thres)
    print("conta: ", conta)
    print("epoch: ", epochs)
    for i in range(1):
        print("cycle:", i)
        # mice_outliers(i)
        ele_outliers(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
