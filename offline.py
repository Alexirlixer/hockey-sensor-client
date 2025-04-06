import time
from measurements import * 
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


def to_label(v):
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
    

if __name__ == "__main__":
    '''
    Do shot detection and classification offline by replaying the measurements
    we have collected in the dataset 
    '''
    dataset = 'test-snaphandle'
    
    model = LSTMClassifier(36, 64, 2, 4)
    model.load_state_dict(torch.load('shots.model'))
    model.eval()

    bucket_size = 250 # in ms 
    window_size = int(3 * 1000/bucket_size) # 3 seconds in 250ms buckets

    measurements = sensor_measurements(f'./data/data-{dataset}.csv')
    buckets = sensor_bucket_max(bucket_size, measurements)
    windows = sensor_detection_window(window_size, buckets)

    count = 0
    pos = 0
    last_peak_timestamp = 0
    start = time.perf_counter()
    for w in windows:
        count += 1
        m = np.argmax(w[:,POS_ACC])

        # predict motion 
        v = w[:,[POS_AX, POS_AY, POS_AZ]].flatten().astype(np.float32)
        pred = torch.argmax(model(torch.from_numpy(np.array([v]))))

        # not unknown and not seen before
        if pred != 0 and w[m, POS_TS] != last_peak_timestamp :
            pos += 1
            last_peak_timestamp = w[m, POS_TS]

            s = mph(w, m, bucket_size/1000)
            print(f"{pos}:[ts:{w[m,POS_TS]}]  [acc: {w[m,POS_ACC]:.2f}] [mph: {s:.2f}] [{to_label(pred)}]")
        
    elapsed = time.perf_counter() - start
    print(f"{count} in {elapsed:.4f} s")