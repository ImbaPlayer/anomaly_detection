import pandas as pd
from datetime import datetime

# inputName = "/data/xgr/sketch_data/equinix-nyc.dirB.20190117-140000.UTC.anon.pcap"

PACKET_NUMBER = 5

def new_extract(num):
    inputName = "/data/sym/one-class-svm/data/univ1/5+mean/packet-level/univ1-50W-{}.csv".format(1)
    saveName = "/data/sym/one-class-svm/data/univ1/5+mean/dec-feature/univ1-50W-{}.csv".format(1)
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
    result_col_names = ["srcIP", "srcPort", "dstIP", "dstPort", "protocol", 
                        "first", "second", "third", "fourth", "fifth",
                        "mean", "var", "min", "max", "flowSize"]
    new_df = pd.DataFrame(columns=result_col_names)
    for key,group in grouped:
        # print(key)
        ori_list = group.iloc[0].values.tolist()[0:-1]
        temp_mean = int(group["length"].mean())
        temp_var = int(group["length"].var(ddof=0))
        temp_min = group["length"].min()
        temp_max = group["length"].max()
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