import numpy as np

def get_ave():
    a = [11349, 5162, 6147, 7324, 6181, 8449, 6101, 5446, 6204]
    b = [5054, 18040, 12826, 16070, 11148, 9233, 12782, 13170, 9754]
    univ1 = [2841, 4455, 4447, 3922, 4463, 4701, 4499, 2918, 2805]
    print("caida-A", np.mean(a))
    print("caida-B", np.mean(b))
    print("univ1", np.mean(univ1))
    # caida-A 6929.222222222223
    # caida-B 12008.555555555555
    # univ1 3894.5555555555557
if __name__ == "__main__":
    # path = "test{}"
    # print(path)
    # print(path.format(1))
    get_ave()