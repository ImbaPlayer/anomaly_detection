#!/bin/sh

# no weight
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 0 no >> torch_log/no_weight/5-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 1 no >> torch_log/no_weight/time-${var}.log 
done


for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 2 no >> torch_log/no_weight/size-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 3 no >> torch_log/no_weight/stat-${var}.log 
done

# weight 
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 0 yes >> torch_log/weight/5-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 1 yes >> torch_log/weight/time-${var}.log 
done


for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 2 yes >> torch_log/weight/size-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/NN/10_NN.py ${var} 3 yes >> torch_log/weight/stat-${var}.log 
done