# Staff Detection Baseline

This baseline solves the evaluation task by detecting people with YOLO, tracking them with ByteTrack, and comparing person crops with the two staff snapshots extracted from the evaluation PDF.

## Setup

Use Python 3.11 or 3.12 for the smoothest PyTorch installation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python detect_staff.py
```

The first run downloads the YOLO model. Results are written to `outputs/annotated.mp4`, `outputs/annotated.avi`, and `outputs/staff_frames.csv`. The CSV contains the frame number, timestamp, presence flag, and staff bounding-box center coordinates. If your browser or Windows video player cannot open the MP4, use `annotated.avi` or open the MP4 with VLC; the MP4 is encoded as `mp4v`, which is not supported by every browser.

To use different snapshots:

```powershell
python detect_staff.py --reference path\to\snapshot1.jpg --reference path\to\snapshot2.jpg
```

The appearance threshold is intentionally exposed because it must be tuned on additional videos. Lower `--match-threshold` to find more candidates; raise it to reduce false positives.