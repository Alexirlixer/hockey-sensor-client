from measurements import * 
import pandas as pd


# create training dataset
if __name__ == "__main__":
    # labels: 0 unk, 1 wrist, 2 snap, 3 slap
    unknown = 0
    datasets = {
        'move': 0,
        'static': 0,
        'test': 0, 
        'radar-wrist': 1,
        '100-wrist': 1,
        'test-wrist': 1, 
        'radar-snap': 2,
        'test-snaphandle': 2,
        'test-snaponly': 2,
        'test-slaponly': 3,
    }

    samples = 0 # total number of samples
    shots = 0 # total number of detected shots

    bucket_size = 250 # in ms 
    window_size = int(3 * 1000/bucket_size) # 3 seconds in 250ms buckets

    # Data frame which will collect all the data we have for training 
    cols = [f"a{i}" for i in range(0,window_size*3)] # 3 acc measurements per window point
    cols.append("label")
    df = pd.DataFrame(columns=cols)
 
    for dataset, label in datasets.items():
        # processs each dataset 
        measurements = sensor_measurements(f'./data/data-{dataset}.csv')
        buckets = sensor_bucket_max(bucket_size, measurements)
        windows = sensor_detection_window(window_size, buckets)

        ts = 0
        # for each window from the dataset 
        for w in windows:
            samples += 1

            m = np.argmax(w[:,POS_ACC])
        
            # try to get only when the peak is somewhere towards the middle 
            # of the window (to help w. ml model)
            if (window_size - m) >= window_size/3: 
                s = mph(w, m, bucket_size/1000)
                t = w[m, POS_TS]

                # try to get only shots over 50mph 
                if s > 50:
                    # never seen before
                    if  ts != t:
                        shots += 1
                        ts = w[m,POS_TS]

                        r = w[:,[POS_AX, POS_AY, POS_AZ]].flatten()
                        r = np.append(r, label)
                        df.loc[samples] = r
                else:
                    r = w[:,[POS_AX, POS_AY, POS_AZ]].flatten()
                    r = np.append(r, unknown)
                    df.loc[samples] = r
                
    print(f"{shots} shots out of {samples} samples")
    df.to_csv('dataset.csv')