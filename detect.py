import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from buffer import fixed_size_buffer

POS_TS = 0
POS_GX = 1
POS_GY = 2
POS_GZ = 3
POS_AX = 4
POS_AY = 5
POS_AZ = 6
POS_ACC = 7


def mph(w, m, dur):
    mps = 2.2369362920544
    acc = w[m,POS_ACC]
    v = np.sqrt((0.385 * (acc * dur)**2)/0.160)
    s = v * mps 
    return s


def shot_type(v):
    match v:
        case 0: 
            return "unknown"
        case 1:        
            return "wrist"
        case 2:
            return "snap"
        case 3:
            return "slap"
        case _:
            return f"{v}"


class shot_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
    

class shot_detect:
    def __init__(self, size=250):
        self._size = size # real time aggregation buffer size 
        self._dim = 8  # measurement array length 
        self._count = 0 # detected shot count 
        # create measurement rolling buffer
        self._buffer = np.zeros((self._size, self._dim))
        self._buffer_pos = 0
        # create measurement sliding window 
        self._window_size = int(3*1000/self._size)
        self._window = fixed_size_buffer(self._window_size,
                                         self._dim)
        # create classifier 
        self._model = shot_classifier(36, 64, 2, 4)
        self._model.load_state_dict(torch.load('shots.model'))
        self._model.eval()
        self._last_peak_timestamp = 0

    def on_measurement(self, data):
        self._process(data)

    def count(self):
        return self._count 

    def _process(self, data):
        # collect measurements for size ms and return the one
        # that has the maximum linear acceleration

        # change timestamp to be in milliseconds and add acceleration
        data[POS_TS] = data[POS_TS] / 1000
        acc = np.sqrt(data[POS_AX]**2 + data[POS_AY]**2 + data[POS_AZ]**2)
        data = np.append(data, acc)
    
        # we expect the data to be stored per milliseconds so 
        # check the difference between the first timestamp and 
        # the current one is not bigger than the size
        if (self._buffer_pos == 0 or 
            (self._buffer_pos < self._size and data[POS_TS] - self._buffer[0][POS_TS] < self._size)):
            # store measurements
            self._buffer[self._buffer_pos] = data
            self._buffer_pos += 1
        else:
            # adding new the new element goes over the window size limit
            # sent it out, reset and store the new one
            # print(f"bucket trigger at {buffer_pos}")
            i = np.argmax(self._buffer[:,POS_ACC])

            # send this to be processed
            self._process_window(self._buffer[i])
            
            # reset buffer
            self._buffer_pos = 0
            self._buffer.fill(0)

            # store measurements
            self._buffer[self._buffer_pos] = data
            self._buffer_pos += 1
    
    def _process_window(self, data):
        self._window.append(data)

        if len(self._window) >= self._window_size:
            self._classify(self._window.get())

    def _classify(self, w):
        m = np.argmax(w[:,POS_ACC])

        # predict motion 
        v = w[:,[POS_AX, POS_AY, POS_AZ]].flatten().astype(np.float32)
        pred = torch.argmax(self._model(torch.from_numpy(np.array([v]))))

        # not unknown and not seen before
        if pred != 0 and w[m, POS_TS] != self._last_peak_timestamp :
            self._count += 1
            self._last_peak_timestamp = w[m, POS_TS]

            s = mph(w, m, self._size/1000)
            print(f"[ts:{w[m,POS_TS]}]  [acc: {w[m,POS_ACC]:.2f}] [mph: {s:.2f}] [{shot_type(pred)}]")

