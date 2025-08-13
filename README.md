## Hockey Shot detection and analysis using real time sensors and machine learning 

This project explores the use of inertial measurement units (IMUs) to analyze and provide real-time feedback on hockey players' shooting techniques.

### Intro 

The core problem addressed is the limitations of current hockey shot tracking methods, which are often inaccurate, inefficient, and fail to provide useful feedback for player development. This project proposes a new tool utilizing IMUs to collect gyroscopic and accelerometer data, simulate the performed motion in real time, and compare it to a pre-established baseline from a skilled player. This comparison aims to help players fine-tune their technique, improving shot speed and accuracy.

### Methodology 

The methodology involves developing a 3D-printed stick cap to house IMU sensors (Adafruit BNO055 accelerometer/gyroscope) powered by a Lithium Ion battery. Data is collected from shooting sessions of different shot types (snapshot, wristshot, slapshot), and this data is used to train an LSTM (Long Short-Term Memory) machine learning model for shot classification and real-time detection. The success of the project is quantified by achieving a shot detection model accuracy above 90 percent.

### Results

The results indicate that the data collection process was successful, and the collected datasets were effectively filtered to reduce noise. Both peak detection algorithms and the LSTM machine learning model proved successful in detecting when a shot occurred and classifying motions. This demonstrates the feasibility of using IMUs to measure and represent swinging motions in real time and provide technique feedback.

### Applications

Beyond hockey, the findings have broader implications for other sports (e.g., golf, tennis, lacrosse, field hockey) and even non-athletic fields requiring precise movements, such as physical therapy monitoring or surgeon training.

For more detailed information, please refer to the original research paper (paper.pdf) included in the repo. 