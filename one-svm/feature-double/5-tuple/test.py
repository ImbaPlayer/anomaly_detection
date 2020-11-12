import pandas as pd
import re

def split_text2(text, length):
    text_arr = re.findall(r'.{%d}' % int(length), text)
    result = [int(data, 2) for data in text_arr]
    return result

def test():
    getBits = lambda bits: lambda n: pd.Series(list(('{0:0%db}'%bits).format(int(n))))
    bits = 16
    num = 255
    print(('{0:0%db}'%bits))
    print(('{0:0%db}'%bits).format(num))
    original = ('{0:0%db}'%bits).format(num)
    print(type(original))
    print(split_text2(original, 8))
    bit_list = split_text2(original, 8)
    print(bit_list)
    getBytes = lambda bits: lambda n: pd.Series(split_text2(('{0:0%db}'%bits).format(int(n)), 8))
    # a = list(('{0:0%db}'%bits).format(num))
    # print(a)
    # b = pd.Series(a)
    # print(b)
    # c = ('{0:0%dB}'%bits).format(num)
    # print(c)
if __name__ == "__main__":
    test()