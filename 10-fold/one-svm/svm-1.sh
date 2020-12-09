#!/bin/bash
# @Author: Guanglin Duan
# @Date:   2020-12-05 00:47:22
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-05 11:49:44
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


# univ2
for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 0 3 >>log/univ2/5-var-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 1 3 >>log/univ2/time-var-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 2 3 >>log/univ2/size-var-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 3 3 >>log/univ2/stat-var-${nu}.log
    done
done


# univ1
for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 0 2 >>log/univ1/5-${var}-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 1 2 >>log/univ1/time-${var}-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 2 2 >>log/univ1/size-${var}-${nu}.log
    done
done

for var in 0.05 0.10 0.15 0.20;do
    for nu in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20;do
    python -u /data/dgl/anomaly_detection/10-fold/one-svm/10_SVM_train_type.py ${var} ${nu} 3 2 >>log/univ1/stat-${var}-${nu}.log
    done
done