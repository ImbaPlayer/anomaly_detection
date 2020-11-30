#!/bin/sh

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_MLP_Predict.py ${var} 0 >> 5-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_MLP_Predict.py ${var} 1 >> time-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_MLP_Predict.py ${var} 2 >> size-${var}.log &
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_MLP_Predict.py ${var} 3 >> stat-${var}.log &
done