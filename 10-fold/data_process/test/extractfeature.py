from scapy.all import *
from scapy.layers.inet import *
from datetime import datetime
import pandas as pd
import sys
import csv
import os


class ExtractFeature:
    def __init__(self):
        self.filter = ['UDP', 'TCP']
        self.flow = {}
        self.flow_max_time = 60
        self.flow_to_cap = 320
        self.packet_feature_name = [
            'Send/Receive indicator',#only for index except 1
            'Interval',#only for index except 1
            'DSCP',
            'TTL',
            'Protocol',
            'Source port',
            'Destination port',
            'ACK',
            'RST',
            'SYN',
            'FIN',
            'PSH',
            'Payload Size'
        ]
        self.packet_dic = {}
        self.last_packet_time = 0
        self.flow_feature = []

    def packet2flow(self, packet):
        if IP not in packet:
            return
        if UDP not in packet and TCP not in packet:
            return
        srcIP = packet[IP].src
        dstIP = packet[IP].dst
        if UDP in packet:
            srcPort = packet[UDP].sport
            dstPort = packet[UDP].dport
        else:
            srcPort = packet[TCP].sport
            dstPort = packet[TCP].dport
        protocol = packet[IP].proto
        Tuple1 = (srcIP, dstIP, srcPort, dstPort, protocol)
        Tuple2 = (dstIP, srcIP, dstPort, srcPort, protocol)
        if self.flow.get(Tuple1) or self.flow.get(Tuple2):
            if self.flow.get(Tuple1):
                if (packet.time - self.flow[Tuple1][-1][0].time) <= self.flow_max_time:
                    self.flow[Tuple1][-1].append(packet)
                else:
                    self.flow[Tuple1].append([packet])
            else:
                if (packet.time - self.flow[Tuple2][-1][0].time) <= self.flow_max_time:
                    self.flow[Tuple2][-1].append(packet)
                else:
                    self.flow[Tuple2].append([packet])
        else:
            self.flow[Tuple1] = [[packet]]

    def get_feature_from_flow(self, flow, Tuple, filename):
        if len(flow) >= self.flow_to_cap:
            for i in range(self.flow_to_cap):
                self.get_feature_from_packet(flow[i], i, self.last_packet_time, Tuple)
                self.flow_feature.append(self.packet_dic)
            self.write_to_csv(filename)

    def get_feature_from_packet(self, packet, number, last_time, Tuple):
        if TCP in packet:
            if number >= 1:
                #self.packet_dic[self.packet_feature_name[0]] = (packet[IP].src == Tuple[0])
                self.packet_dic[self.packet_feature_name[1]] = packet.time - last_time
            self.packet_dic[self.packet_feature_name[2]] = packet[IP].tos
            self.packet_dic[self.packet_feature_name[3]] = packet[IP].ttl
            self.packet_dic[self.packet_feature_name[4]] = packet[IP].proto
            self.packet_dic[self.packet_feature_name[5]] = packet[TCP].sport
            self.packet_dic[self.packet_feature_name[6]] = packet[TCP].dport
            TCP_FLAG = packet[TCP].flags
            if 'A' in TCP_FLAG:
                self.packet_dic[self.packet_feature_name[7]] = 1
            else:
                self.packet_dic[self.packet_feature_name[7]] = -1
            if 'R' in TCP_FLAG:
                self.packet_dic[self.packet_feature_name[8]] = 1
            else:
                self.packet_dic[self.packet_feature_name[8]] = -1
            if 'S' in TCP_FLAG:
                self.packet_dic[self.packet_feature_name[9]] = 1
            else:
                self.packet_dic[self.packet_feature_name[9]] = -1
            if 'F' in TCP_FLAG:
                self.packet_dic[self.packet_feature_name[10]] = 1
            else:
                self.packet_dic[self.packet_feature_name[10]] = -1
            if 'P' in TCP_FLAG:
                self.packet_dic[self.packet_feature_name[11]] = 1
            else:
                self.packet_dic[self.packet_feature_name[11]] = -1
            if Raw in packet:
                self.packet_dic[self.packet_feature_name[12]] = len(packet[Raw])
            else:
                self.packet_dic[self.packet_feature_name[12]] = 0

        else:
            if number >= 1:
                #self.packet_dic[self.packet_feature_name[0]] = (packet[IP].src == Tuple[0])
                self.packet_dic[self.packet_feature_name[1]] = packet.time - last_time
            self.packet_dic[self.packet_feature_name[2]] = packet[IP].tos
            self.packet_dic[self.packet_feature_name[3]] = packet[IP].ttl
            self.packet_dic[self.packet_feature_name[4]] = packet[IP].proto
            self.packet_dic[self.packet_feature_name[5]] = packet[UDP].sport
            self.packet_dic[self.packet_feature_name[6]] = packet[UDP].dport
            self.packet_dic[self.packet_feature_name[7]] = 0
            self.packet_dic[self.packet_feature_name[8]] = 0
            self.packet_dic[self.packet_feature_name[9]] = 0
            self.packet_dic[self.packet_feature_name[10]] = 0
            self.packet_dic[self.packet_feature_name[11]] = 0
            if Raw in packet:
                self.packet_dic[self.packet_feature_name[12]] = len(packet[Raw])
            else:
                self.packet_dic[self.packet_feature_name[12]] = 0
        self.last_packet_time = packet.time

    def write_to_csv(self, filename):
        with open(filename, 'a', newline='') as csvfile:
            csvw = csv.writer(csvfile)
            onerow = []
            for i in range(self.flow_to_cap): 
                onerow= onerow + list(self.flow_feature[i].values())
            csvw.writerow(onerow)

def main():
    EF = ExtractFeature()
    filename = "/data/xgr/sketch_data/caida_dirA/equinix-nyc.dirA.20190117-130000.UTC.anon.pcap"
    sniff(offline=filename, prn=EF.packet2flow, store=0, count=500000)
    for key in EF.flow:
        for i in range(len(EF.flow[key])):
            EF.get_feature_from_flow(EF.flow[key][i], key, 'test.csv')

if __name__ == "__main__":
    a = datetime.now()
    print("start time", a)

    main()
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)



