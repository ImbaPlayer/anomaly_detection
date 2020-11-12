import pandas as pd
getBits = lambda bits: lambda n: pd.Series(list(('{0:0%db}'%bits).format(int(n))))
# print(pd.DataFrame([0]).apply(getBits(4)))
print(pd.DataFrame([10]).apply(getBits(4)))

a = int(255) / 255.0
print(type(a))