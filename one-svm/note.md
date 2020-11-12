### one-class-svm

```
# 从原始pcap文件中提取数据包五元组，包长等信息
python pcap2csv.py --pcap H:\exp_data\pcap_data\univ1_all.pcap --csv original-univ1.cs

# 提取出每条流的五元组，以及统计特征
python csv_process.py

# 将特征转为二进制
python dec2bin.py
```

