# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-12-06 00:55:44
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-06 01:13:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.datasets
import pandas as pd
import numpy as np
import sys
import math
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from torch.autograd import Variable
from datetime import datetime
from sklearn.model_selection import KFold

# elePercent train_type weight_type dataset
elePercent = float(sys.argv[1])
train_type_num = int(sys.argv[2])
weight_type = sys.argv[3]
data_type_num = int(sys.argv[4])
rng = np.random.RandomState(10)
PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1", "univ2", "unibs"]
ALL_TRAIN_TYPE = ["5-tuple", "time", "size", "stat"]
server_name = "dgl"
dataSetType = ALL_DATA_TYPE[data_type_num]
trainType = ALL_TRAIN_TYPE[train_type_num]

#define the network class
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # self.fc1 = nn.Linear(198, 80)
        self.fc1 = nn.Linear(387, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        # x = torch.tanh(x)
        return x

    def predict(self, x):
        pred = torch.tanh(self.forward(x))
        # print("pred", pred)
        ans = []
        for t in pred:
            if t[0] > 0:
                ans.append(1)
            else:
                ans.append(-1)
        return torch.tensor(ans)

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

def main(num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X, yc = load_data(dataSetType, trainType, num)

    # 10 fold validation
    KF = KFold(n_splits=10, shuffle=True, random_state=10)
    report_list_nn = []
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = yc[train_index], yc[test_index]
    
        mice_count = sum(y_train==-1)
        elephant_count = sum(y_train==1)
        # better weight
        # elephant_percent = (elephant_count + mice_count) / elephant_count
        # weight_alpha = int(math.log10(elephant_percent))
        # weight_alpha = math.pow(10, weight_alpha)
        # elephant_weight = weight_alpha * (elephant_count + mice_count) / elephant_count

        # reweight
        elephant_weight = (elephant_count + mice_count) / elephant_count
        print("elephant_weight", elephant_weight)
        print("original mice count: ", sum(yc==-1))
        print("original elephant count: ", sum(yc==1))

        smote = SMOTE(random_state=10)
        # X_train_sample, y_train_sample = smote.fit_sample(X_train, y_train)
        X_train_sample, y_train_sample = X_train, y_train
        print(sum(y_train==1), sum(y_train==-1), sum(y_test==1), sum(y_test==-1))
        print("sampling:", sum(y_train_sample==1), sum(y_train_sample==-1))
        
        X_train_sample = torch.from_numpy(X_train_sample).type(torch.FloatTensor)
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
        y_train_sample = torch.tensor(y_train_sample.values).type(torch.FloatTensor)

        torch_dataset = Data.TensorDataset(X_train_sample, y_train_sample)
        # 把 dataset 放入 DataLoader
        BATCH_SIZE = 200
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,               # 要不要打乱数据
            num_workers=2,              # 多线程来读数据
        )
        # y_test = torch.tensor(y_test.values).type(torch.FloatTensor).to(device)
        #neural network
        model = MyNetwork()
        model.fc1 = nn.Linear(X_train_sample.shape[1], 80)
        model.to(device)
        #define loss function
        # criterion = nn.CrossEntropyLoss()
        class_weight = Variable(torch.FloatTensor([1, elephant_weight, 1])).to(device)
        # criterion = nn.BCEWithLogitsLoss(weight=class_weight[y_train_weight.long()])
        #define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        #number of epochs
        # epochs = int(sys.argv[3])
        epochs = 50
        #list to store losses
        losses = []
        for i in range(epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = model.forward(batch_x)
                y_pred = y_pred.squeeze()
                #compute cross entrophy loss
                if weight_type == "no":
                    # print("no weight")
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.BCEWithLogitsLoss(weight=class_weight[batch_y.long()])
                loss = criterion(y_pred, batch_y)
                #add loss to the list
                losses.append(loss.item())
                #clear the previous gradients
                optimizer.zero_grad()
                #compute gradients
                loss.backward()
                #adjust weights
                optimizer.step()

        #predict
        predictions = model.predict(X_test)
        predictions = predictions.cpu()
        # print(accuracy_score(model.predict(X),y))
        c_matrix = confusion_matrix(y_test, predictions)
        print(c_matrix)
        temp_report = classification_report(y_test, predictions, output_dict=True)
        report_list_nn.append(temp_report)
        print(classification_report(y_test,predictions))
    final_report = get_avg_report(report_list_nn)
    print("final report nn", final_report)

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
    for i in range(3):
        print("cycle:", i)
        # mice_outliers(i)
        main(i)
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)