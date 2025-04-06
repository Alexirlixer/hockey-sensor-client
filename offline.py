import time
import pandas as pd
from detect import *


def sensor_replay(path):
    detector = shot_detect()

    cols = ['ts', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    df = pd.read_csv(path, names=cols)
  
    for index, row in df.iterrows():
        detector.on_measurement(row.to_numpy())

    return detector.count()


if __name__ == "__main__":
    '''
    Do shot detection and classification offline by replaying the measurements
    we have collected in the dataset 
    '''
    dataset = 'test-snaphandle'

    start = time.perf_counter()
    count = sensor_replay(f"./data/data-{dataset}.csv")
    elapsed = time.perf_counter() - start
    print(f"{count} in {elapsed:.4f} s")