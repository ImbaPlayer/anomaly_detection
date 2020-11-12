import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 5

def new_extract(num):
    inputName = "/data/sym/one-class-svm/data/online/packet-level/caida-A-50W-{}.csv".format(num)
    # inputName = "original.csv"
    saveName = "/data/sym/one-class-svm/data/online/dec-feature/caida-A-50W-{}.csv".format(num)
    # saveName = "1.csv"
    #指定分隔符为/t
    # time srcIP srcPort dstIP dstPort protocol length
    col_names = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", 
    "ip_ihl", "ip_tos", "ip_flags", "ip_ttl", "tcp_dataofs", "tcp_flag", "tcp_window", "udp_len", "length"]
    df = pd.read_csv(inputName, delimiter="|", names=col_names)
    # print(df)
    # print(df)
    mask = (df["protocol"]=="TCP") | (df["protocol"]=="IPv4") | (df["protocol"]=="UDP")
    tcp = df[mask]
    tcp = tcp.drop(["time"], axis=1)

    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    # print(grouped["length"])
    #计算数据包的数量
    tcp["flowSize"] = grouped["length"].transform(sum)

    grouped=tcp.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])
    result = grouped.head(1)
    result.to_csv(saveName, index=False)
    #df = pd.read_csv(saveName)
    #print(df)

if __name__ == '__main__':
    a = datetime.now()
    print("start time", a)
    for i in range(11,12):
        new_extract(i)
        print("finish", i)
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)
