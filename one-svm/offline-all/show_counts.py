import pandas as pd
from datetime import datetime

def get_counts(filePath, col_names):
    df = pd.read_csv(filePath, delimiter="|", names=col_names)
    print(df.shape)
    return df

if __name__ == "__main__":
    filePath1 = "/data/sym/one-class-svm/data/offline-5/packet-level/caida-A-50W-{}.csv".format(0)
    filePath2 = "/data/sym/one-class-svm/data/offline-all/packet-level/caida-A-50W-{}.csv".format(0)
    col_names1 = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    col_names2 = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "ip_tos", "ip_flags", "ip_ttl", "tcp_flag", "tcp_window", "length"]
    df1 = get_counts(filePath1, col_names1)
    df2 = get_counts(filePath2, col_names2)

    count = 0
    for i in range(10000):
        if df1.iloc[i]['length'] != df2.iloc[i]['length']:
            print("df1", df1.iloc[i]['length'])
            print("df2", df2.iloc[i]['length'])
            count += 1
    
    print("count", count)