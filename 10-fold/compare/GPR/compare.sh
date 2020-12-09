#!/bin/sh
# @Author: Guanglin Duan
# @Date:   2020-12-02 15:44:13
# @Last Modified by:   Guanglin Duan
# @Last Modified time: 2020-12-02 17:35:05

# var 5-tuple GPR univ1
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 0 1 2 >> log/univ1/5-GPR-${var}.log 
done

# var 5-tuple NB univ1
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 0 2 2 >> log/univ1/5-NB-${var}.log 
done

# var 5-tuple SVM univ1
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 0 3 2 >> log/univ1/5-SVM-${var}.log 
done

# var 5-tuple DT univ1
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 0 4 2 >> log/univ1/5-DT-${var}.log 
done
# -----------------------------------------------------------------------------------------------------------
# time univ1 
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 1 1 2 >> log/univ1/time-GPR-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 1 2 2 >> log/univ1/time-NB-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 1 3 2 >> log/univ1/time-SVM-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 1 4 2 >> log/univ1/time-DT-${var}.log 
done

# -----------------------------------------------------------------------------------------------------------
# size univ1 
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 2 1 2 >> log/univ1/size-GPR-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 2 2 2 >> log/univ1/size-NB-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 2 3 2 >> log/univ1/size-SVM-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 2 4 2 >> log/univ1/size-DT-${var}.log 
done

# -----------------------------------------------------------------------------------------------------------
# stat univ1 
for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 3 1 2 >> log/univ1/stat-GPR-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 3 2 2 >> log/univ1/stat-NB-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 3 3 2 >> log/univ1/stat-SVM-${var}.log 
done

for var in 0.2 0.15 0.1 0.05;do
python /data/dgl/anomaly_detection/10-fold/compare/GPR/10_compare.py ${var} 3 4 2 >> log/univ1/stat-DT-${var}.log 
done
# -----------------------------------------------------------------------------------------------------------
