#### 文件说明：

##### 提取dec-feature

`get_stat_transform.py`：提取5-tuple以及前n个包的stat特征

`get_size_transform.py`：提取前n个包的大小

`get_time_transform.py`：提取前n个包的时间间隔

##### 提取bin-feature

`bin_stat.py`：5-tuple, stat

`bin_size.py`： 前n个包的大小



#### pcap2csv

优化想法：对于大于10个包的，直接pass