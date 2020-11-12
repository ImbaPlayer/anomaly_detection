#!/bin/sh

#for var in 0 1 2 3 4 5 6 7 8 9;do
#python ../offline-all/pcap2csv_add.py --pcap /data/xgr/sketch_data/caida_dirA/equinix-nyc.dirA.20190117-130${var}00.UTC.anon.pcap --csv ../data/offline-all/packet-level/caida-A-50W-${var}.csv
#done

python ../offline-all/csv_process_all.py

python ../offline-all/dec2bin_all.py