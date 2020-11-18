# -*- coding: utf-8 -*-
# @Author: Guanglin Duan
# @Date:   2020-11-15 12:11:20
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-11-15 12:13:35
import argparse
import os.path
import sys

from scapy.utils import RawPcapReader, PcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP

filePath = "H:\\exp_data\\pcap_data\\caida-50w-1.pcap"
filePath2 = "H:\\exp_data\\pcap_data\\caida-100w.pcap"
filePath3 = "H:\\exp_data\\pcap_data\\equinix-nyc.dirB.20190117-130000.UTC.anon.pcap"
filePath4 = "H:\\exp_data\\pcap_data\\univ1_all.pcap"
filePath5 = "/data/xgr/sketch_data/caida_dirA/equinix-nyc.dirA.20190117-130000.UTC.anon.pcap"
TCP_FLAG_MAP = {'U':1, 'A':2, 'P':3, 'R':4, 'S':5, 'F':6}
print(TCP_FLAG_MAP["S"])
def getLen():
    temp_len = []
    for data in RawPcapReader(filePath3):
        i += 1
        # print(data.show())
        # print(len(data[0]))
        temp_len.append(len(data[0]))
        if i > 2000:
            break
        if len(data[0]) == 64:
            print(IP(data[0]).show())
    print(max(temp_len))
    print(min(temp_len))    
    print(list(set(temp_len)))
if __name__ == "__main__":
    i = 0
    for data in RawPcapReader(filePath5):
        i += 1
        ip_pkt_sc = IP(data[0])
        # print(data[0])
        # print(data[0][6] >> 5)
        print(ip_pkt_sc.show())
        print(data[1])
        

        if i > 10:
            break
    