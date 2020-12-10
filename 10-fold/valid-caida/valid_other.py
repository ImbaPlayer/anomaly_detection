# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-03 17:09:05
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-02 17:20:33

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import sys

# sys params: ele_percent train_type clf_type dataset_type
# type: float int int int
elePercent = float(sys.argv[1])
train_type_num = int(sys.argv[2])
clf_type_num = int(sys.argv[3])
data_type_num = int(sys.argv[4])
rng = np.random.RandomState(10)
PACKET_NUMBER = 10
ALL_TRAIN_TYPE = ["5-tuple", "time", "size", "stat"]
ALL_CLF_TYPE = ["NN", "GPR", "Naive_Bayes", "SVM", "DT"]
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1", "univ2", "unibs"]
server_name = "sym"
dataSetType = ALL_DATA_TYPE[data_type_num]
trainType = ALL_TRAIN_TYPE[train_type_num]
clfType = ALL_CLF_TYPE[clf_type_num]
TOTAL_FLOW_COUNT = 50000

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
        # get head total_flow_count
        dfb = dfb.head(TOTAL_FLOW_COUNT)
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
        # get head total_flow_count
        df = df.head(TOTAL_FLOW_COUNT)
        dfb = dfb.head(TOTAL_FLOW_COUNT)

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

def get_clf(clf_type):
    if clf_type == "GPR":
        clf = GaussianProcessRegressor(n_restarts_optimizer=10, random_state=10)
    elif clf_type == "Naive_Bayes":
        clf = GaussianNB()
    elif clf_type == "SVM":
        kernel_type = ['linear', 'poly', 'rbf']
        clf = svm.SVC(kernel='linear', gamma='scale')
    elif clf_type == "DT":
        tree_criterions = ['gini','entropy']
        clf = DecisionTreeClassifier(criterion = "gini", random_state=10)
    elif clf_type == "NN":
        clf = MLPClassifier(hidden_layer_sizes=(100, 40), activation='tanh', max_iter=400, random_state=10)
    return clf

def ele_outliers(num):
    
    
    X, yc = load_data(dataSetType, trainType, num)

    # 10 fold validation
    KF = KFold(n_splits=10, shuffle=True, random_state=10)
    report_list_nn = []
    
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = yc[train_index], yc[test_index]

        # undersample
        # smote = RandomUnderSampler(random_state=10)
        # X_train_sample, y_train_sample = smote.fit_sample(X_train, y_train)
        X_train_sample, y_train_sample = X_train, y_train

        print(sum(y_train==1), sum(y_train==-1), sum(y_test==1), sum(y_test==-1))
        print("sampling:", sum(y_train_sample==1), sum(y_train_sample==-1))

        clf = get_clf(clfType)
        clf.fit(X_train_sample, y_train_sample)
        predictions = clf.predict(X_test)
        if clfType == "GPR":
            predictions[predictions > 0] = 1
            predictions[predictions < 0] = -1
        c_matrix = confusion_matrix(y_test, predictions)
        # print(c_matrix)
        temp_report = classification_report(y_test, predictions, output_dict=True)
        report_list_nn.append(temp_report)
        # print(classification_report(y_test,predictions))

    final_report = get_avg_report(report_list_nn)
    print("final report {}".format(clfType), final_report)
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
    print("dataset", dataSetType)
    print("elePercent:", elePercent)
    print("train type", trainType)
    print("clf type", clfType)
    print("total flow count", TOTAL_FLOW_COUNT)
    for i in range(1):
        print("cycle:", i)
        # mice_outliers(i)
        ele_outliers(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
