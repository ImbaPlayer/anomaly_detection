import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 5

def new_extract(num):
    inputName = "/data/sym/one-class-svm/data/count/packet-level/caida-A-50W-{}.csv".format(num)
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

    tcp['count'] = grouped["length"].transform(len)
    # ip.len
    tcp['flowSize'] = grouped["length"].transform(sum)
    first = grouped.head(1)
    length = first['count']
    ip_flow_size = first['flowSize']
    for i in range(200):
        print(first.iloc[i]['count'], first.iloc[i]['flowSize'])

    max_length = length.max()
    min_length = length.min()
    mean_length = length.mean()
    print("max length", max_length)
    print(min_length)
    print(mean_length)
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