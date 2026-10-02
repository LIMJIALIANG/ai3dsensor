# Staff Detection Baseline

This baseline solves the evaluation task by detecting people with YOLO, tracking them with ByteTrack, and comparing each person's upper torso with the two staff snapshots extracted from the evaluation PDF. The green-shirt image in the PDF explains the tag; the two overhead images are the actual staff references. The supplied clip contains one staff member, so the strongest stable match is selected and all other people are ignored.

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

The first run downloads the YOLO model. Results are written to `outputs/annotated.mp4` and `outputs/staff_frames.csv`. The CSV contains one row per input frame. Each row includes the ByteTrack ID, presence flag, staff count (`0` or `1`), and staff bounding-box center coordinates.

To use a different staff-tag snapshot:

```powershell
python detect_staff.py --reference path\to\staff_tag_snapshot.jpg
```

The detector uses a `0.60` appearance threshold, eight-frame persistence, and a recent movement requirement to reject people standing near tables. Lower `--match-threshold` or `--stable-frames` only when testing shows missed staff detections; raise them to reduce false positives. Because the name tag is very small in the overhead video, production-level tag-only accuracy should be improved with labeled crops of the tag (`staff_tag` versus `no_staff_tag`) and a small classifier trained on those crops.