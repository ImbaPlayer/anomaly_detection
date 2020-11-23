# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-16 19:00:38
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-16 19:14:53
import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 10
ALL_DATA_TYPE = ["caida-A", "caida-B", "univ1"]

def new_extract(num):
    data_type = ALL_DATA_TYPE[0]
    inputName = "/data/sym/anomaly_detection/data/10-fold/{}/packet-level/{}-{}.csv".format(data_type, data_type, num)
    saveName = "/data/sym/anomaly_detection/data/10-fold/{}/dec-feature/{}-{}.csv".format(data_type, data_type, num)
    # inputName = "/data/sym/one-class-svm/data/mean_of_five/packet-level/univ1-50W-{}.csv".format(num)
    # saveName = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/univ1-50W-{0}-{1}.csv".format(PACKET_NUMBER, num)
    # inputName = "../original.csv"
    # saveName = "size-1.csv"
    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    tcp = tcp.drop(["time"], axis=1)
    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    result_col_names = ["srcIP", "srcPort", "dstIP", "dstPort", "protocol"]
    result_col_names.extend(["pkt-size-{}".format(i) for i in range(PACKET_NUMBER)])
    result_col_names.extend(["mean", "var", "min", "max", "flowSize"])
    # result_col_names = ["srcIP", "srcPort", "dstIP", "dstPort", "protocol", 
    #                     "first", "second", "third", "fourth", "fifth","sixth","seventh","eighth","nineth","tenth",
    #                     "mean", "var", "min", "max", "flowSize"]
    new_df = pd.DataFrame(columns=result_col_names)
    for key,group in grouped:
        # print(key)
        ori_list = group.iloc[0].values.tolist()[0:-1]
        # get first n packet from every group
        n_packets = group.head(PACKET_NUMBER)
        temp_mean = int(n_packets["length"].mean())
        temp_var = int(n_packets["length"].var(ddof=0))
        temp_min = n_packets["length"].min()
        temp_max = n_packets["length"].max()
        temp_len = group["length"].sum()
        # print("statistics", [temp_mean, temp_var, temp_min, temp_max, temp_len])
        packet_length = group["length"].values.tolist()
        N_length = []
        if len(packet_length) >= PACKET_NUMBER:
            N_length = packet_length[0: PACKET_NUMBER]
        else:
            for _ in range(PACKET_NUMBER - len(packet_length)):
                packet_length.append(0)
            N_length = packet_length
        # print("N_length", N_length)
        N_length.extend([temp_mean, temp_var, temp_min, temp_max, temp_len])
        ori_list.extend(N_length)
        # print(N_length)
        temp_df = pd.DataFrame([ori_list], columns=result_col_names)
        new_df = new_df.append(temp_df, ignore_index=True)
    print(new_df.shape)
    new_df.to_csv(saveName, index=False)
    #df = pd.read_csv(saveName)
    #print(df)
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