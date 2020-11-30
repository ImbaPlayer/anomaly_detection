#!/bin/sh

for var in 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train.py ${var} ${var} >> 5-${var}-${var}.log
done
