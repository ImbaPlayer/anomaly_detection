import pandas as pd
from datetime import datetime

def get_counts(filePath, col_names):
    df = pd.read_csv(filePath, delimiter="|", names=col_names)
    print(df.shape)
    return df

def get_max(filePath):
    df1 = pd.read_csv(filePath)
    length = df1["flowSize"]
    max_length = length.max()
    min_length = length.min()
    mean_length = length.mean()
    print("max length", max_length)
    print(min_length)
    print(mean_length)

if __name__ == "__main__":
    filePath1 = "/data/sym/one-class-svm/data/5+mean/dec-feature/caida-A-50W-{}.csv".format(0)
    filePath2 = "/data/sym/one-class-svm/data/offline-all-len/dec-feature/caida-A-50W-{}.csv".format(0)
    filePath3 = "/data/sym/one-class-svm/data/count/dec-feature/caida-A-50W-{}.csv".format(0)
    col_names1 = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "length"]
    col_names2 = ["time", "srcIP", "srcPort", "dstIP", "dstPort", "protocol", "ip_tos", "ip_flags", "ip_ttl", "tcp_flag", "tcp_window", "length"]
    # df1 = get_counts(filePath1, col_names1)
    # df2 = get_counts(filePath2, col_names2)

    get_max(filePath3)
    # for i in range(df1.shape[0]):
    #     temp = df1.iloc[i]["flowSize"]
    #     if temp > 20000:
    #         print(temp)

