# AI Bicep Curl Validator & Counter 🏋️‍♂️

A real-time computer vision application that counts bicep curl repetitions and validates exercise form using MediaPipe Pose Estimation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📝 Description

This project utilizes the **MediaPipe Pose Landmarker** to track human body keypoints in real-time. Unlike simple repetition counters, this system implements **form validation logic** to ensure the user is performing the exercise correctly.

It calculates the angle of the elbow joint to count repetitions and monitors the vertical position of the elbow (using Exponential Moving Average smoothing) to detect "elbow drift"—a common mistake where the lifter uses momentum or shoulder activation instead of the bicep.

## ✨ Key Features

* **Repetition Counting:** Automates counting based on joint angles (Down > 160°, Up < 60°).
* **Form Analysis:** Detects invalid reps if the elbow drifts vertically or the upper arm swings significantly.
* **Signal Smoothing:** Implements Exponential Moving Average (EMA) to reduce keypoint jitter and provide stable metrics.
* **Real-time Visualization:** Draws the skeleton, repetition count, and live form feedback (OK/BAD) on the video feed.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Infin-Nine/AI-Bicep-Curl-Validator.git
    cd AI-Bicep-Curl-Validator
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Model**
    You need the MediaPipe Pose Landmarker model.
    - Download `pose_landmarker_full.task` from [Google MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models).
    - Place the file in the root directory of this project.

## 🚀 Usage

Run the main script using Python:

```bash
python main.py
