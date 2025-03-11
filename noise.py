import pandas as pd
import plotly

dfs = pd.read_csv("data-static.csv", names = ["ts", "gx", "gy", "gz", "ax", "ay", "az"])
dfm = pd.read_csv("data-move.csv", names = ["ts", "gx", "gy", "gz", "ax", "ay", "az"])

print("calculate normal distribution for noise")
names = ["gx", "gy", "gz", "ax", "ay", "az"]
ranges = {}
for name in names:
    m = float(dfs[name].mean())
    s = float(dfs[name].std())
    ranges[name] = (m-3*s, m+3*s)

print("noise normal distribution ranges for each measurement")
print(ranges)

print('measuring effectiveness against static')
for name in names:
    low, high = ranges[name]
    nf = dfs[(dfs[name] > high) | (dfs[name] < low)]
    print(f"{name}: total {len(dfs)} vs left {len(nf)}")

print("compare how many measurements are left from the motion if we filter out noise")
for name in names:
    low, high = ranges[name]
    nf = dfm[(dfm[name] > high) | (dfm[name] < low)]
    print(f"{name}: total {len(dfm)} vs left {len(nf)}")

print("rolling average test:")

st = dfs.rolling(100).mean()
fig = plotly.plot(st, "line")
fig.show()
