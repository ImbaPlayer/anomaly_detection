#!/bin/sh

for var in 1 2 3 4 5 6 7 8 9;do
python /data/sym/one-class-svm/mean_of_five/pcap2csv.py --pcap /data/xgr/sketch_data/caida_dirA/equinix-nyc.dirA.20190117-130${var}00.UTC.anon.pcap --csv /data/sym/one-class-svm/data/mean_of_five/packet-level/caida-A-50W-${var}.csv
done

#python csv_process.py

#python dec2bin.py