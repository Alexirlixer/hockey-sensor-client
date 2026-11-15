## Hockey Shot detection and analysis using real time sensors and machine learning

This project explores the use of inertial measurement units (IMUs) to analyze and provide real-time feedback on hockey players' shooting techniques.

### Intro

The core problem addressed is the limitations of current hockey shot tracking methods, which are often inaccurate, inefficient, and fail to provide useful feedback for player development. This project proposes a new tool utilizing IMUs to collect gyroscopic and accelerometer data, simulate the performed motion in real time, and compare it to a pre-established baseline from a skilled player. This comparison aims to help players fine-tune their technique, improving shot speed and accuracy.

### Methodology

The methodology involves developing a 3D-printed stick cap to house IMU sensors (Adafruit BNO055 accelerometer/gyroscope) powered by a Lithium Ion battery. Data is collected from shooting sessions of different shot types (snapshot, wristshot, slapshot), and this data is used to train an LSTM (Long Short-Term Memory) machine learning model for shot classification and real-time detection. The success of the project is quantified by achieving a shot detection model accuracy above 90 percent.

### Results

The results indicate that the data collection process was successful, and the collected datasets were effectively filtered to reduce noise. Both peak detection algorithms and the LSTM machine learning model proved successful in detecting when a shot occurred and classifying motions. This demonstrates the feasibility of using IMUs to measure and represent swinging motions in real time and provide technique feedback.

### Applications

Beyond hockey, the findings have broader implications for other sports (e.g., golf, tennis, lacrosse, field hockey) and even non-athletic fields requiring precise measurements, such as physical therapy monitoring or surgeon training.

For more detailed information, please refer to the original research paper (`paper.pdf`) included in the repo.

### Architecture

This is the desktop/ML half of a two-part system. The stick-mounted sensor itself — MicroPython firmware that reads the IMU and streams measurements over Bluetooth LE — lives in [hockey-sensor](https://github.com/Alexirlixer/hockey-sensor). This repo connects to that BLE peripheral, processes the measurement stream, and runs shot detection/classification.

### How it works

1. **Ingest** — `collect.py` and `live.py` scan for and connect to the `hockey-sensor` BLE peripheral (`bleak`), subscribing to its notify characteristic for gyro + linear-acceleration samples.
2. **Buffer & detect** — each incoming sample is pushed into a rolling buffer (`buffer.py`) inside `shot_detect` (`detect.py`). A peak-detection pass over acceleration magnitude flags candidate shot events.
3. **Classify** — detected windows are run through a trained LSTM (`shots.model`, defined in `detect.py`) to classify the shot as wrist, snap, or slap, and `mph()` estimates shot speed from the peak acceleration.
4. **Train offline** — `dataset.py` builds labeled training windows from the recorded CSVs in `data/`, and `train.ipynb` trains the LSTM used by `detect.py`. `offline.py` replays a recorded session through the same detection pipeline for evaluation without live hardware.

### Repo layout

- `collect.py` — connect to the sensor and log raw measurements to CSV.
- `live.py` — connect to the sensor and run real-time shot detection/classification.
- `offline.py` — replay a recorded CSV through the detection pipeline.
- `detect.py` — the shot-detection buffer and LSTM classifier.
- `dataset.py` / `train.ipynb` — build the labeled training set and train the model.
- `*.ipynb` — exploratory analysis notebooks (acceleration, gyro, noise, shot speed).
- `data/` — recorded sensor sessions used for training and evaluation.
- `shots.model` — the trained classifier used by `detect.py`.
- `paper.pdf` — the full write-up of methodology and results.
