# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-14 21:57:59
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-17 15:38:32
import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]
def new_extract(num):
    data_type = ALL_DATA_TYPE[0]
    # inputName = "/data/sym/one-class-svm/data/mean_of_five/packet-level/caida-B-50W-{}.csv".format(num)
    inputName = "/data/sym/anomaly_detection/data/10-fold/{}/packet-level/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/sym/anomaly_detection/data/10-fold/{}/time-dec-feature/{}-{}.csv".format(data_type, data_type, num)
    # inputName = "/data/sym/one-class-svm/data/mean_of_five/packet-level/univ1-50W-{}.csv".format(num)
    # saveName = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(PACKET_NUMBER, num)
    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    # tcp = tcp.drop(["time"], axis=1)
    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    result_col_names = ["interval-{}".format(i) for i in range(PACKET_NUMBER - 1)]
    result_col_names.append("flowSize")
    
    new_df = pd.DataFrame(columns=result_col_names)
    for key,group in grouped:
        ori_list = group.iloc[0].values.tolist()[0:-1]
        temp_len = group["length"].sum()
        # get first n packet from every group
        n_packets = group.head(PACKET_NUMBER)
        arrive_time = n_packets["time"].values.tolist()
        time_interval = get_interval(arrive_time)
        # put flow size in the end
        time_interval.append(temp_len)
        temp_df = pd.DataFrame([time_interval], columns=result_col_names)
        new_df = new_df.append(temp_df, ignore_index=True)
    print(new_df.shape)
    # print(new_df)
    new_df.to_csv(saveName, index=False)
    # df = pd.read_csv(saveName)
    #print(df)

    # test
    # dfb = new_df.loc[:, ["interval-{}".format(i) for i in range(3)]]
    # print(dfb)

def get_interval(arrive_time):
    time_interval = []
    for i in range(len(arrive_time) - 1):
        time_interval.append(arrive_time[i+1] - arrive_time[i])
    for _ in range(PACKET_NUMBER - 1 - len(time_interval)):
        time_interval.append(0)
    return time_interval
if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(1):
        new_extract(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)