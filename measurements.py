import numpy as np
import pandas as pd
from buffer import fixed_size_buffer

POS_TS = 0
POS_GX = 1
POS_GY = 2
POS_GZ = 3
POS_AX = 4
POS_AY = 5
POS_AZ = 6
POS_ACC = 7

def sensor_measurements(path):
    cols = ['ts', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    df = pd.read_csv(path, names=cols)
  
    for index, row in df.iterrows():
        yield row.to_numpy()


def sensor_bucket_max(size, measurements):
    # collect measurements for size ms and return the one
    # that has the maximum linear acceleration
    dimensions = 8
    buffer = np.zeros((size, dimensions))
    buffer_pos = 0

    for data in measurements:
        # change timestamp to be in milliseconds and add acceleration
        data[POS_TS] = data[POS_TS] / 1000
        acc = np.sqrt(data[POS_AX]**2 + data[POS_AY]**2 + data[POS_AZ]**2)
        data = np.append(data, acc)
    
        # we expect the data to be stored per milliseconds so 
        # check the difference between the first timestamp and 
        # the current one is not bigger than the size
        if buffer_pos == 0 or (buffer_pos < size and data[POS_TS] - buffer[0][POS_TS] < size):
            # store measurements
            buffer[buffer_pos] = data
            buffer_pos += 1
        else:
            # adding new the new element goes over the window size limit
            # sent it out, reset and store the new one
            # print(f"bucket trigger at {buffer_pos}")
            i = np.argmax(buffer[:,POS_ACC])
            yield buffer[i]
            
            # reset buffer
            buffer_pos = 0
            buffer.fill(0)

             # store measurements
            buffer[buffer_pos] = data
            buffer_pos += 1


def sensor_detection_window(size, buckets):
    dimensions = 8
    buffer = fixed_size_buffer(size, dimensions)

    for data in buckets:
        buffer.append(data)

        if len(buffer) >= size:
            yield buffer.get()

def mph(w, m, dur):
    mps = 2.2369362920544
    acc = w[m,POS_ACC]
    v = np.sqrt((0.385 * (acc * dur)**2)/0.160)
    s = v * mps 
    return s
