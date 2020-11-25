import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.datasets
import pandas as pd
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

def my_loss_function(x, y):
    
    return torch.mean(torch.pow((x - y), 2))
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]
ALL_TRAIN_TYPE = ["5-tuple", "time", "size", "stat"]
dataSetType = ALL_DATA_TYPE[0]
trainType = ALL_TRAIN_TYPE[3]
# fileName1 = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/caida-A-50W-5-{}.csv".format(0)
# fileName2 = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/caida-A-50W-5-{}.csv".format(0)
fileName1 = "/data/sym/anomaly_detection/data/10-fold/{}/dec-stat/{}-{}.csv".format(dataSetType, dataSetType, 0)
fileName2 = "/data/sym/anomaly_detection/data/10-fold/{}/bin-stat/{}-{}.csv".format(dataSetType, dataSetType, 0)
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(fileName1)
    dfb = pd.read_csv(fileName2)
    print("shape", dfb.shape)
    
    #csv to tensor
    # train = pd.read_csv('train.csv')
    # train_tensor = torch.tensor(train.values)
    
    #conver to matrix
    X = dfb.values
    X[X=='0'] = -1
    X[X=='1'] = 1
    yr = df['flowSize']

    thres = int(sys.argv[1])
    # weight_alpha = int(sys.argv[2])
    print("thres: ", thres)
    # thres = 250
    yc = yr.copy(deep=True)
    yc[yr > thres] = 1
    yc[yr <= thres ] = -1
    
    mice_count = sum(yc==-1)
    elephant_count = sum(yc==1)
    elephant_percent = (elephant_count + mice_count) / elephant_count
    weight_alpha = int(math.log10(elephant_percent))
    weight_alpha = math.pow(10, weight_alpha)
    elephant_weight = weight_alpha * (elephant_count + mice_count) / elephant_count

    # reweight
    elephant_weight = (elephant_count + mice_count) / elephant_count
    print("elephant_weight", elephant_weight)
    print("original mice count: ", sum(yc==-1))
    print("original elephant count: ", sum(yc==1))

    # X = torch.from_numpy(X).type(torch.FloatTensor)
    # print(yc)
    # yc = torch.tensor(yc.values).type(torch.FloatTensor)

    #oversampling by smote
    #test train split
    X_train, X_test, y_train, y_test = train_test_split(X, yc, random_state=10)

    # y_train_weight = y_train.copy(deep=True)
    # y_train_weight[y_train_weight > 0] = 1
    # y_train_weight[y_train_weight <= 0] = 0
    # y_train_weight = torch.tensor(y_train_weight.values).type(torch.LongTensor).to(device)

    #oversampling minority class
    #while(sum(y_train==-1) / sum(y_train==1) > 2):
         #mask = (y_train == 1)
         #X_train = np.concatenate((X_train, X_train[mask]), axis=0)
         #y_train = np.concatenate((y_train, y_train[mask]), axis=0)
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
    model.fc1 = nn.Linear(dfb.shape[1], 80)
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
        # y_pred = model.forward(X_train_sample)
        # y_pred = y_pred.squeeze()
        # # print(y_pred)
        # # pred = torch.Tensor(y_pred).type(torch.FloatTensor)
        # # print(y_pred.shape)
        # # print(y.shape)
        # #compute cross entrophy loss
        # loss = criterion(y_pred, y_train_sample)
        # #add loss to the list
        # losses.append(loss.item())
        # #clear the previous gradients
        # optimizer.zero_grad()
        # #compute gradients
        # loss.backward()
        # #adjust weights
        # optimizer.step()
        # if (i % 100) == 0:
        #     print("Epoch %d, Loss: %.4f" % (i+1, loss.item()))

    #predict
    predictions = model.predict(X_test)
    predictions = predictions.cpu()
    # print(accuracy_score(model.predict(X),y))
    c_matrix = confusion_matrix(y_test, predictions)
    print(c_matrix)
    print(classification_report(y_test,predictions))
    
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)

    
    main()
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)