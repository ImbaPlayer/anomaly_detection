#!/bin/sh
flowCount=5W

# ele trainType GPR caida-A
for ele in 0.05 0.1 0.15 0.2;do
    for trainType in 0 1 2 3;do
    python -u /data/sym/anomaly_detection/10-fold/valid-caida/valid_other.py ${ele} ${trainType} 1 0 >> /data/sym/anomaly_detection/10-fold/valid-caida/log/${flowCount}/${ele}-${trainType}-GPR.log
    done
done

# NB
for ele in 0.05 0.1 0.15 0.2;do
    for trainType in 0 1 2 3;do
    python -u /data/sym/anomaly_detection/10-fold/valid-caida/valid_other.py ${ele} ${trainType} 2 0 >> /data/sym/anomaly_detection/10-fold/valid-caida/log/${flowCount}/${ele}-${trainType}-NB.log
    done
done

# NB
for ele in 0.05 0.1 0.15 0.2;do
    for trainType in 0 1 2 3;do
    python -u /data/sym/anomaly_detection/10-fold/valid-caida/valid_other.py ${ele} ${trainType} 3 0 >> /data/sym/anomaly_detection/10-fold/valid-caida/log/${flowCount}/${ele}-${trainType}-SVM.log
    done
done

# DT
for ele in 0.05 0.1 0.15 0.2;do
    for trainType in 0 1 2 3;do
    python -u /data/sym/anomaly_detection/10-fold/valid-caida/valid_other.py ${ele} ${trainType} 4 0 >> /data/sym/anomaly_detection/10-fold/valid-caida/log/${flowCount}/${ele}-${trainType}-SVM.log
    done
done