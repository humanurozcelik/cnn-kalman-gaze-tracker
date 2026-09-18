# Webcam Gaze Tracking with CNN + Kalman Filtering

This is a webcam-based gaze-tracking project I developed while working on computer vision and eye-tracking methods.

The main idea is to use both eye regions from a webcam frame, train a small CNN during calibration, and then smooth the predicted screen coordinates with a Kalman filter.

## How it works

1. OpenCV captures webcam frames.
2. MediaPipe Face Mesh finds facial and eye landmarks.
3. Left and right eye regions are cropped and resized.
4. The two eye images are combined into a `64 × 64 × 2` input.
5. A 9-point calibration is used to collect training samples.
6. A small CNN learns to map eye appearance to screen coordinates.
7. A Kalman filter smooths the predicted gaze position.
8. The session is saved to CSV for later analysis.

## Files

- `main.py` — runs calibration, training, tracking, and CSV logging
- `gaze_analysis.py` — calculates error metrics and creates plots
- `requirements.txt` — Python dependencies

## Setup

The current version is intended for Python 3.11 on Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Windowed mode is easier for testing:

```powershell
python main.py --windowed --preview
```

Press **ENTER** to start calibration and follow the target with your eyes.

After the session, the default output is:

```text
lab_gaze_tracking_data.csv
```

To analyze the recorded session:

```powershell
python gaze_analysis.py --csv lab_gaze_tracking_data.csv
```

The analysis script reports spatial error statistics and creates trajectory, density, and error plots. If `fastdtw` is installed, it can also calculate a DTW-based trajectory metric.

## Notes

This is still an experimental student project. Results depend strongly on lighting, webcam position, head movement, calibration quality, and screen setup.

The post-processing alignment is only used to compare trajectories after a session; it is not a direct measurement of system latency.
