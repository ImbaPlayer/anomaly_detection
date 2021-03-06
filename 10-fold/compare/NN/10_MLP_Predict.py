# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-03 17:09:05
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-24 15:44:42

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import KFold
from datetime import datetime
import pandas as pd
import numpy as np
import sys

# thres = int(sys.argv[1])
elePercent = float(sys.argv[1])
train_type_num = int(sys.argv[2])
rng = np.random.RandomState(10)
PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1", "univ2"]
ALL_TRAIN_TYPE = ["5-tuple", "time", "size", "stat"]
server_name = "dgl"
dataSetType = ALL_DATA_TYPE[2]
trainType = ALL_TRAIN_TYPE[train_type_num]

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

def get_file_name(trainType):
    path1 = ""
    path2 = ""
    if trainType == "5-tuple":
        path1 =  "/data/{}/anomaly_detection/data/10-fold/{}/dec-stat/{}-{}.csv"
        path2 =  "/data/{}/anomaly_detection/data/10-fold/{}/bin-5/{}-{}.csv"
    elif trainType == "size":
        path1 =  "/data/{}/anomaly_detection/data/10-fold/{}/dec-size/{}-{}.csv"
        path2 =  "/data/{}/anomaly_detection/data/10-fold/{}/bin-size/{}-{}.csv"
    elif trainType == "stat":
        path1 =  "/data/{}/anomaly_detection/data/10-fold/{}/dec-stat/{}-{}.csv"
        path2 =  "/data/{}/anomaly_detection/data/10-fold/{}/bin-stat/{}-{}.csv"
    return path1, path2

def load_data(dataSetType, trainType, num):
    if trainType == "time":
        # user time interval as features
        fileName1 = "/data/{}/anomaly_detection/data/10-fold/{}/dec-time/{}-{}.csv".format(server_name, dataSetType, dataSetType, num)
        dfb = pd.read_csv(fileName1)
        yr = dfb['flowSize']
        
        # drop flowSize
        dfb = dfb.drop(["time","srcIP","srcPort","dstIP","dstPort","protocol","flowSize"], axis=1)
        # convert to matrix
        X = dfb.values
        
    else:
        path1, path2 = get_file_name(trainType)
        fileName1 = path1.format(server_name, dataSetType, dataSetType, num)
        fileName2 = path2.format(server_name, dataSetType, dataSetType, num)
        df = pd.read_csv(fileName1)
        dfb = pd.read_csv(fileName2)

        # get specific cols
        # dfb = dfb.loc[:, get_col_names(trainType)]
        
        #conver to matrix
        X = dfb.values
        X[X=='0'] = -1
        X[X=='1'] = 1
        yr = df['flowSize']

        # thres = int(sys.argv[1])
        
    yc = yr.copy(deep=True)
    thres = get_thres(yr, elePercent)
    print("thres: ", thres)
    yc[yr <= thres] = -1
    yc[yr > thres ] = 1
    print("original mice count: ", sum(yc==-1))
    print("original elephant count: ", sum(yc==1))
    return X, yc
def ele_outliers(num):
    
    
    X, yc = load_data(dataSetType, trainType, num)

    # 10 fold validation
    KF = KFold(n_splits=10, shuffle=True, random_state=10)
    report_list_nn = []
    report_list_forest = []
    
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = yc[train_index], yc[test_index]

        # undersample
        # smote = RandomUnderSampler(random_state=10)
        # X_train_sample, y_train_sample = smote.fit_sample(X_train, y_train)
        X_train_sample, y_train_sample = X_train, y_train

        print(sum(y_train==1), sum(y_train==-1), sum(y_test==1), sum(y_test==-1))
        print("sampling:", sum(y_train_sample==1), sum(y_train_sample==-1))
        #neural network
        # print("neural network:")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 40), activation='tanh', max_iter=400, random_state=10)
        mlp.fit(X_train_sample, y_train_sample)
        predictions = mlp.predict(X_test)
        c_matrix = confusion_matrix(y_test, predictions)
        # print(c_matrix)
        temp_report = classification_report(y_test, predictions, output_dict=True)
        report_list_nn.append(temp_report)
        # print(classification_report(y_test,predictions))

        #random forest
        # print("random forest:")
        rf = RandomForestClassifier(n_estimators=30, class_weight={1:1,-1:1}, random_state=10)
        rf = rf.fit(X_train_sample, y_train_sample)
        predictions = rf.predict(X_test)
        c_matrix = confusion_matrix(y_test, predictions)
        # print(c_matrix)
        temp_report = classification_report(y_test, predictions, output_dict=True)
        report_list_forest.append(temp_report)
        # print(classification_report(y_test,predictions))


    final_report = get_avg_report(report_list_nn)
    print("final report nn", final_report)
    final_report = get_avg_report(report_list_forest)
    print("final report random forest", final_report)
def get_avg_report(report_list):
    report_array = np.array(report_list)
    # np.save('NN-5-1.npy', report_array)

    
    report_list_0 = []
    report_list_1 = []
    acc_list = []
    result = {}
    for data in report_list:
        report_list_0.append(data['-1'])
        report_list_1.append(data['1'])
        acc_list.append(data['accuracy'])
    df_0 = pd.DataFrame(report_list_0)
    df_1 = pd.DataFrame(report_list_1)
    result['-1'] = dict(df_0.mean())
    result['1'] = dict(df_1.mean())
    result["accuracy"] = np.mean(acc_list)
    # np.save("NN-5-2.npy", result)
    return result      
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    print("elePercent:", elePercent)
    for i in range(1,20):
        print("cycle:", i)
        # mice_outliers(i)
        ele_outliers(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
