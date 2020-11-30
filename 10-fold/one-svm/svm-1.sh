#!/bin/sh

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${var} 1 >> time-${var}-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${var} 2 >> size-${var}-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${var} 3 >> stat-${var}-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
    for nu in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
    python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 1 >> time-${var}-${nu}.log &
    done
done

for nu in 0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py 0.2 ${nu} 3 >> log/stat-0.2-${nu}.log
done

for nu in 0.01 0.05 0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py 0.15 ${nu} 3 >> log/stat-0.15-${nu}.log
done

for nu in 0.01 0.05 0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py 0.10 ${nu} 3 >> log/stat-0.10-${nu}.log
done

for nu in 0.01 0.05 0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do
python /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py 0.05 ${nu} 3 >> log/stat-0.05-${nu}.log
done