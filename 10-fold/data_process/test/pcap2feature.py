import argparse
import os.path
import sys

from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from datetime import datetime
from scapy.all import *
from scapy.layers.inet import *

PACKET_NUMBER = 10

class FeatureExtract:
    def __init__(self, save_path):
        self.save_path = save_path
        self.flow_dict = {}
        self.flow_max_time

    def render_csv_row(self, pkt_sc):
        """
        process packet one by one
        """
        print(type(packet))
        print(packet)

        
        # ether_pkt_sc = Ether(pkt_sc)
        # if ether_pkt_sc is None:
        #     return False

        # if not ether_pkt_sc.haslayer(IP):
        #     return False
        # ip_pkt_sc = ether_pkt_sc[IP]       # <<<< Assuming Ethernet + IPv4 here
        
        ip_pkt_sc = IP(pkt_sc)
        if ip_pkt_sc.version != 4:
            return False
        proto = ip_pkt_sc.fields['proto']
        print("protocol", pkt_sc[IP].proto)
        if proto == 17 and ip_pkt_sc.haslayer('UDP'):
            udp_pkt_sc = ip_pkt_sc[UDP]
            l4_payload_bytes = bytes(udp_pkt_sc.payload)
            l4_proto_name = 'UDP'
            l4_sport = udp_pkt_sc.sport
            l4_dport = udp_pkt_sc.dport
        elif proto == 6 and ip_pkt_sc.haslayer('TCP'):
            tcp_pkt_sc = ip_pkt_sc[TCP]
            l4_payload_bytes = bytes(tcp_pkt_sc.payload)
            l4_proto_name = 'TCP'
            l4_sport = tcp_pkt_sc.sport
            l4_dport = tcp_pkt_sc.dport
        else:
            # Currently not handling packets that are not UDP or TCP
            # print('Ignoring non-UDP/TCP packet')
            return False
        srcIP = ip_pkt_sc.src
        dstIP = ip_pkt_sc.dst
        # length = len(pkt_sc)
        length = ip_pkt_sc.len
        # pkt_time = timeInfo.sec + timeInfo.usec / (10**6)
        pkt_time = pkt_sc.time
        # pkt_time = 0
        # print("pkt_time", pkt_time)

        # Each line of the CSV has this format
        fmt = '{0}|{1}|{2}|{3}|{4}|{5}|{6}'
        # time srcIP srcPort dstIP dstPort protocol length

        print(fmt.format(pkt_time,                # {0}
                     srcIP,                   # {1}
                     l4_sport,                # {2}
                     dstIP,                   # {3}
                     l4_dport,                # {4}
                     l4_proto_name,           # {5}
                     length),                 # {6}
          file=self.fh_csv)
        
        tuple_5 = (srcIP, dstIP, l4_proto_name, l4_sport, l4_dport)
        if self.flow_dict.get(tuple_5):
            # if (pkt_time - self.flow_dict[tuple_5]["last_time"]) <= self.flow_max_time:
            if self.flow_dict[tuple_5]["count"] > PACKET_NUMBER:
                return
            else:
                self.flow_dict[tuple_5]["pkt-size"].append(length)
                self.flow_dict[tuple_5]["stat"] = self.get_stat(self.flow_dict[tuple_5]["stat"], length)
                self.flow_dict[tuple_5]["interval"].append(pkt_time - self.flow_dict[tuple_5]["last_time"])
                self.flow_dict[tuple_5]["count"] += 1
                self.flow_dict[tuple_5]["last_time"] = pkt_time
        else:
            self.flow_dict[tuple_5] = {"pkt-size":[length],
                                        "stat":[length, length, length],# max, min, sum
                                        "interval":[],
                                        "count":1,
                                        "last_time":pkt_time}
        return True
    def get_stat(self, stat_list, new_length):
        temp_max = max(stat_list[0], new_length)
        temp_min = min(stat_list[1], new_length)
        temp_sum = stat_list[2] + new_length
        return [temp_max, temp_min, temp_sum]
        

def main():
    save_path = "test-pcap2f.csv"
    FE = FeatureExtract(save_path)
    # filename = "/data/xgr/sketch_data/caida_dirA/equinix-nyc.dirA.20190117-130000.UTC.anon.pcap"
    filename = "H:\exp_data\pcap_data\caida-50w-1.pcap"
    with open(save_path, 'w') as FE.fh_csv:
        sniff(offline=filename, prn=FE.render_csv_row, store=0, count=5)
    # for key in EF.flow:
    #     for i in range(len(EF.flow[key])):
    #         EF.get_feature_from_flow(EF.flow[key][i], key, 'test.csv')

if __name__ == "__main__":
    a = datetime.now()
    print("start time", a)

    main()
    
    b = datetime.now()
    print("end time", b)
    durn = (b-a).seconds
    print("duration", durn)