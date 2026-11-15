# hockey-sensor-client

Real-time hockey shot detection from a sensor in the stick. A Raspberry Pi Pico W and a BNO055 IMU inside the stick's end cap stream gyroscope and linear-acceleration samples over Bluetooth LE. This repository holds the laptop side. It receives that stream, compresses it into a fixed-size view of the last three seconds of motion, and runs a small LSTM that decides whether a shot just happened and whether it was a wrist, snap or slap shot. It also estimates the shot's speed from the peak acceleration. The repo also contains the recorded sessions, the training pipeline, the analysis notebooks and the research paper behind the project.

| Repository | Role |
| --- | --- |
| [hockey-sensor](https://github.com/Alexirlixer/hockey-sensor) | Firmware on the stick: IMU sampling, on-device noise gate, binary packing, BLE peripheral |
| **hockey-sensor-client** (this repo) | Laptop side: BLE client, data collection, shot detection and classification, speed estimate, training |

A write-up of the whole project is on the portfolio site: [alexirlixer.github.io](https://alexirlixer.github.io/#hockey). The full research write-up is [`paper.pdf`](paper.pdf), by Alex Apostolescu and Erik Kittelson (Engineering Senior Research, Academy of Engineering and Technology).

## How it works

```mermaid
flowchart LR
    subgraph STICK["Stick cap (hockey-sensor)"]
        IMU["BNO055 IMU"] --> PICO["Pico W<br/>noise gate + pack"]
    end
    PICO -- "BLE notify 0xBEEF<br/>32-byte #60;Qffffff" --> BLEAK["bleak client<br/>live.py / collect.py"]
    BLEAK --> BUCKET["250 ms buckets<br/>keep max |a| sample"]
    BUCKET --> WINDOW["sliding window<br/>12 buckets = 3 s"]
    WINDOW --> LSTM["LSTM<br/>36 inputs → 4 classes"]
    LSTM --> OUT["shot type<br/>+ speed (mph)"]
    BLEAK -. "collect.py" .-> CSV["CSV sessions<br/>data/"]
    CSV -. "dataset.py" .-> DS["dataset.csv"]
    DS -. "train.ipynb" .-> MODEL["shots.model"]
    MODEL -.-> LSTM
```

1. **Connect.** `live.py` (and `collect.py`) use [bleak](https://github.com/hbldh/bleak) to scan for a device advertising service `0xBEEF`. It then connects and subscribes to notifications. If the sensor disconnects, the script waits 10 s and scans again.
2. **Decode.** Each notification holds one sample, which is decoded with `struct.unpack("<Qffffff", data)`: a microsecond timestamp, gyro x/y/z and linear acceleration x/y/z. The format is described under [BLE wire format](#ble-wire-format).
3. **Bucket.** `shot_detect` in `detect.py` converts the timestamp to milliseconds and adds a ninth column holding the acceleration magnitude `|a|`. It collects samples into a 250 ms bucket. A bucket closes when a sample arrives 250 ms or more after the bucket's first sample, or when it holds 250 samples. The bucket is then reduced to a single row: the sample with the largest `|a|`.
4. **Window.** Those one-per-bucket rows go into a 12-slot ring buffer (`buffer.py`), which holds three seconds of motion. Once the buffer is full, every new bucket produces a window covering the latest 12 buckets.
5. **Classify.** The x, y and z acceleration of the 12 rows are flattened into 36 numbers and fed to the LSTM, which predicts one of `0` no shot, `1` wrist, `2` snap or `3` slap.
6. **Report.** If the prediction is a shot, and the peak in this window is not the same peak already reported, the shot counter goes up. The script then prints one line per shot:
   ```
   [<count>] [at:<peak timestamp, ms>] [acc: <peak |a|, m/s²>] [mph: <speed estimate>] [<wrist|snap|slap>]
   ```

## BLE wire format

This is the contract with the [hockey-sensor](https://github.com/Alexirlixer/hockey-sensor) firmware.

| Item | Value |
| --- | --- |
| Advertised name | `hockey-sensor` |
| Service UUID | `0000beef-0000-1000-8000-00805f9b34fb` (16-bit `0xBEEF`) |
| Measurement characteristic | `0000beef-0000-1000-8000-00805f9b34fb`, notify, one sample per notification |
| Battery characteristic | `0000b00f-0000-1000-8000-00805f9b34fb`, notify once per second, `<f` volts |

The service and the measurement characteristic share a UUID. That is why `start_notify(SENSOR_SERVICE_UUID, ...)` subscribes to the right characteristic.

Measurement payload: `<Qffffff`, 32 bytes, little-endian.

| Offset | Size | Type | Field | Unit |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | `uint64` | timestamp (`time.ticks_us()` on the Pico) | µs |
| 8 | 12 | 3 × `float32` | gyro x, y, z | deg/s |
| 20 | 12 | 3 × `float32` | linear acceleration x, y, z (gravity removed) | m/s² |

The firmware only sends samples whose linear-acceleration magnitude is above 0.20 m/s², so the stream contains motion only and has no fixed rate. In the recorded sessions the median gap between samples is 4 to 7 ms. `collect.py` writes each decoded sample as one CSV row in the same order: `ts, gx, gy, gz, ax, ay, az`.

## Engineering highlights

### Turning a variable-rate stream into fixed model input

An LSTM needs input of a fixed shape, but the sensor stream has no fixed rate. The firmware drops quiet samples, and the BLE link delivers notifications in bursts. Max-pooling into 250 ms buckets turns this stream into a regular series of four rows per second, and the pooling uses the Pico's own timestamps, so BLE delivery jitter does not affect it. Keeping the **maximum**-magnitude sample, not the mean, matters here. A shot is a short spike lasting tens of milliseconds, and averaging it with 250 ms of follow-through would flatten it. Twelve buckets with three axes each give the 36-value input. Because the firmware drops quiet stretches, no empty buckets are produced: a window is 12 consecutive buckets of *motion*, and idle time between shots is skipped, not padded with zeros.

### Sliding window with peak de-duplication

A new window is classified after every bucket, so a single shot passes through up to 12 consecutive windows. `shot_detect` records the device timestamp of the peak in each window it reports as a shot. It ignores later windows whose peak is that same sample, so one shot is counted once even though it is seen many times.

### One pipeline for live, replay and training

`live.py` and `offline.py` both send samples into the same `shot_detect.on_measurement()`. One receives them from BLE notifications, the other from a CSV file. Replaying a recorded session therefore runs exactly the code a live session runs, with no hardware, as fast as the CPU allows (`offline.py` prints the elapsed time). The training set was built with the same bucket-max and window logic, so the model sees the same input at training time as at inference time.

### From peak detection to a classifier

The first approach was signal processing. `short-speed.ipynb` groups samples into the same 250 ms bucket maxima and runs `scipy.signal.find_peaks` with a 20 m/s² height threshold:

![Peak detection on bucketed acceleration and gyro data from data-test-snaponly.csv](docs/peak-detection.png)

*Peak detection on `data-test-snaponly.csv`, top panel acceleration magnitude (m/s²), lower panels gyro x/y/z (deg/s); figure 4 in the paper.*

This picks out the shots clearly. The next step was to tell *which* shot was taken, and to separate shots from the smaller 20 to 30 m/s² stickhandling peaks between them. The paper describes replacing peak detection with an LSTM that does both jobs in one pass: the "no shot" class handles detection, and the other three classes handle classification.

### Labels from session structure and physics

`dataset.py` labels the data with no manual annotation:

- Each recording session contains a single shot type, so the file name supplies the class (`radar-snap` is snap, `100-wrist` is wrist, `move` and `static` are no shot, and so on).
- A window counts as a shot only if its peak bucket converts to more than 50 mph with the speed formula below, which works out to about 57.6 m/s². Windows whose peak is weaker become **no-shot** examples, even when they come from a shot session. This is how the model learns to ignore stickhandling: `test-snaphandle` mixes snap shots with stickhandling between them.
- A shot is labeled once, at the first window where its peak has moved back to bucket 8 or earlier (buckets are numbered 0 to 11, oldest first). Windows where the peak is still in the newest three buckets are skipped. As a result, almost every positive example has the spike at bucket 8, about three-quarters of a second before the end of the window, so the shot sits in a consistent position.

The paper also describes enlarging the dataset by shifting values while preserving the shape of each shot. In `train.ipynb` this is done by multiplying the whole dataset by 10 random factors drawn from [-10, 10], so it grows to 11 times its original size.

### The model

`shot_classifier` is a 2-layer `nn.LSTM` (36 inputs, 64 hidden units) followed by a `Linear(64, 4)`. That is 59,652 parameters, and the saved `shots.model` is 242 KB. It is trained for 10 epochs with Adam (lr 1e-3), cross-entropy loss and batch size 10. It is small enough to run once per bucket on a laptop CPU with no noticeable delay. Each window enters the network as one flattened 36-value step, so the timing of the shot is captured by the order of the features rather than by the LSTM's time axis.

### Speed from acceleration alone

`mph()` estimates puck speed from the peak bucket with a kinetic-energy-transfer model:

```
v_stick = a_peak × 0.25 s                       # peak acceleration held over one bucket
v_puck  = sqrt(0.385 × v_stick² / 0.160)        # ½·m₁·v₁² = ½·m₂·v₂²
mph     = v_puck × 2.2369
```

A peak of 68 m/s² gives about 59 mph. The speed comes straight from the stick's own motion data, with no radar gun in the loop.

### Hardware that leaves the stick unchanged

![Design sketch of the stick cap: 3D-printed chassis holding the Pico, BNO055 and battery, inserted into the top of the stick](docs/stick-cap-sketch.jpg)

*Design sketch of the stick cap (figure 1 in the paper).*

The electronics sit in a 3D-printed cap made of TPU, a light, flexible filament. It replaces the stick's normal end cap. The cap is longer than a stock one so it can hold the parts, but the extra length goes inside the shaft, so the stick's playing length does not change and it could still be used in a game. A lithium-ion battery powers the system.

## Research context

**Problem.** Shot tracking in hockey usually relies on external speed radars, placed in the goal or around the rink. Other players moving nearby interfere with their readings, and they only report speed, so they tell a player nothing about technique. Studies using inertial sensors to analyze golf and tennis swings suggested that a low-cost IMU on the stick could capture a hockey shot as well.

**Approach.** Build the stick-cap sensor, record shooting sessions of one shot type each (wrist, snap, slap), and train an LSTM to detect and classify shots in real time. The success criterion set in the paper was a model accuracy above 90%.

**Results.** Data collection worked as planned, and the filtered acceleration traces showed one clear peak per shot (figure 2 in the paper). Peak detection reliably marked each shot, and the trained LSTM, run over a sliding window, detected and classified shots in real time. The labeled dataset behind the model holds 1,527 three-second windows across four classes.

**Applications.** The same capture-and-compare idea applies to other swing sports (golf, tennis, lacrosse, field hockey). It could also apply to work that depends on precise movement, such as monitoring physical-therapy exercises, fall detection and surgical training.

## Running it

### Requirements

Python 3.10 or later (`detect.py` uses `match`) and a Bluetooth LE adapter. Packages, taken from the imports:

```sh
pip install bleak numpy pandas torch                      # live.py, collect.py, offline.py, dataset.py
pip install jupyter matplotlib seaborn scipy plotly       # notebooks
```

Run the scripts from the repository root, because `detect.py` loads `shots.model` from the current directory. The sensor must be running the [hockey-sensor](https://github.com/Alexirlixer/hockey-sensor) firmware.

### Live detection

```sh
python live.py
```

Power on the stick. The script finds `hockey-sensor`, connects, and prints a line for each shot it detects. It reconnects automatically if the link drops.

### Recording a session

```sh
python collect.py
```

This logs every sample to `data.csv` and prints it to the console. The file is reopened (and overwritten) each time the script connects, so rename it after each session, for example `mv data.csv data/data-my-session.csv`.

### Replaying a recording offline

```sh
python offline.py
```

This replays `data/data-test-snaphandle.csv` through the live pipeline, then prints each detected shot and the total time taken. To replay a different session, change the `dataset` variable at the bottom of `offline.py`.

### Retraining

1. Add a session to the `datasets` dictionary in `dataset.py`, mapping its name to a label (`0` no shot, `1` wrist, `2` snap, `3` slap).
2. Run `python dataset.py` to rebuild `dataset.csv`.
3. Run all cells in `train.ipynb`. The last cell writes `shots.model`, which `detect.py` loads.

The bundled `dataset.csv` and `shots.model` already match each other and are ready to use; these steps are for when you add or change data.

## Repository layout

| Path | Purpose |
| --- | --- |
| `live.py` | BLE client and real-time shot detection |
| `collect.py` | BLE client that records samples to `data.csv` |
| `offline.py` | Replays a recorded CSV through `shot_detect` |
| `detect.py` | `shot_detect` (bucketing, windowing, de-duplication), `shot_classifier` (the LSTM), `mph()` |
| `buffer.py` | `fixed_size_buffer`, a NumPy ring buffer holding the 12-bucket window |
| `dataset.py` | Builds labeled windows from `data/` into `dataset.csv` |
| `dataset.csv` | 1,527 labeled windows (36 features and a label) |
| `train.ipynb` | Augmentation, train/test split, training and evaluation; writes `shots.model` |
| `shots.model` | Trained LSTM weights (PyTorch `state_dict`) |
| `short-speed.ipynb` | Bucket-max peak detection and per-shot speed estimates |
| `acceleration-analysis.ipynb`, `gyro-analysis.ipynb`, `analyze-raw-noise.ipynb` | Exploratory plots: acceleration magnitude, 3D gyro trajectory, static-vs-moving noise distributions |
| `data/` | Recorded sessions (see below) |
| `docs/` | Figures used in this README |
| `paper.pdf` | Research paper |

### Recorded sessions

| File | Label in `dataset.py` | Length |
| --- | --- | ---: |
| `data-100-wrist.csv` | wrist | 500 s |
| `data-radar-wrist.csv` | wrist (radar readings in `data-radar-shots.txt`) | 126 s |
| `data-test-wrist.csv` | wrist | 34 s |
| `data-radar-snap.csv` | snap (radar readings in `data-radar-shots.txt`) | 114 s |
| `data-test-snaponly.csv` | snap | 67 s |
| `data-test-snaphandle.csv` | snap, with stickhandling between shots | 56 s |
| `data-test-slaponly.csv` | slap | 12 s |
| `data-move.csv` | no shot (general movement) | 64 s |
| `data-static.csv` | no shot (stick at rest, almost entirely below the 0.20 m/s² gate) | 65 s |
| `data-stick.csv` | general stick motion (extra recording) | 201 s |
