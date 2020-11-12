import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 5

def new_extract(num):
    inputName = "/data/sym/one-class-svm/data/count/packet-level/caida-A-50W-{}.csv".format(0)
    saveName = "/data/sym/one-class-svm/data/count/dec-feature/caida-A-50W-{}.csv".format(11)
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
    tcp["flowSize"] = grouped["length"].transform(len)
    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    tcp = grouped.head(1)

    print(tcp.shape)
    tcp.to_csv(saveName, index=False)
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