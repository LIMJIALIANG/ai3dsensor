"""Detect and track the staff member in a 3D-sensor video.

The detector finds people with YOLO and uses appearance similarity against
the tagged-shirt reference snapshot to identify the single staff member.
It produces an annotated video and one CSV row per input frame.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Reference:
    histogram: object
    image: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("sample/sample.mp4"))
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        dest="references",
        help="Staff snapshot. Repeat this option for multiple snapshots.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.60,
        help="Appearance similarity threshold from 0 to 1.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="Consecutive matching frames required before staff_present is true.",
    )
    parser.add_argument(
        "--minimum-movement",
        type=float,
        default=35.0,
        help="Minimum recent center movement in pixels for a staff candidate.",
    )
    return parser.parse_args()


def default_references() -> list[Path]:
    reference_dir = Path("sample/reference_images")
    return [
        reference_dir / "page_1_image_2.jpg",
        reference_dir / "page_1_image_3.jpg",
    ]


def make_histogram(cv2, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram


def appearance_score(cv2, crop, reference: Reference) -> float:
    resized = cv2.resize(crop, (96, 160))
    histogram_score = cv2.compareHist(
        make_histogram(cv2, resized), reference.histogram, cv2.HISTCMP_CORREL
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.resize(reference_gray, (96, 160))
    pixel_score = 1.0 - float(cv2.norm(gray, reference_gray, cv2.NORM_L2)) / (
        255.0 * (96 * 160) ** 0.5
    )
    return max(0.0, min(1.0, 0.65 * ((histogram_score + 1.0) / 2.0) + 0.35 * pixel_score))


def torso_crop(cv2, person_crop):
    """Return the upper torso, where the staff tag is visible."""
    height, width = person_crop.shape[:2]
    top = int(height * 0.18)
    bottom = int(height * 0.72)
    left = int(width * 0.15)
    right = int(width * 0.85)
    return person_crop[top:bottom, left:right]


def load_references(cv2, paths: list[Path]) -> list[Reference]:
    references = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read reference image: {path}")
        references.append(Reference(make_histogram(cv2, cv2.resize(image, (96, 160))), image))
    if not references:
        raise FileNotFoundError(
            "No reference images found. Pass --reference path/to/staff_snapshot.jpg."
        )
    return references


def main() -> None:
    args = parse_args()

    import cv2
    from ultralytics import YOLO

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    reference_paths = args.references or default_references()
    references = load_references(cv2, reference_paths)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / "annotated.mp4"
    output_csv = args.output_dir / "staff_frames.csv"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create MP4 output video")
    model = YOLO(args.model)
    track_stability: dict[int, int] = {}
    track_positions: dict[int, tuple[float, float]] = {}
    track_movement: dict[int, float] = {}

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_number",
            "timestamp_seconds",
            "staff_present",
            "staff_count",
            "track_id",
            "x",
            "y",
            "score",
        ])

        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=args.confidence,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            candidates = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.int().cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else list(range(len(boxes)))
                for track_id, box in zip(ids, boxes):
                    left, top, right, bottom = box
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(width, right), min(height, bottom)
                    if right <= left or bottom <= top:
                        continue
                    crop = torso_crop(cv2, frame[top:bottom, left:right])
                    score = max(appearance_score(cv2, crop, ref) for ref in references)
                    center = ((left + right) / 2, (top + bottom) / 2)
                    track_id = int(track_id)
                    previous_center = track_positions.get(track_id)
                    movement = 0.0
                    if previous_center is not None:
                        movement = ((center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2) ** 0.5
                    track_positions[track_id] = center
                    track_movement[track_id] = min(500.0, track_movement.get(track_id, 0.0) * 0.85 + movement)
                    candidates.append((track_id, score, left, top, right, bottom))

            for track_id, score, *_ in candidates:
                if score >= args.match_threshold:
                    track_stability[track_id] = track_stability.get(track_id, 0) + 1
                else:
                    track_stability[track_id] = max(0, track_stability.get(track_id, 0) - 1)

            staff_candidates = [
                candidate for candidate in candidates
                if candidate[1] >= args.match_threshold
                and track_stability.get(candidate[0], 0) >= args.stable_frames
                and track_movement.get(candidate[0], 0.0) >= args.minimum_movement
            ]
            staff = max(staff_candidates, key=lambda candidate: candidate[1], default=None)
            staff_present = staff is not None
            if staff is not None:
                track_id, score, left, top, right, bottom = staff
                x, y = round((left + right) / 2, 1), round((top + bottom) / 2, 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
                cv2.putText(frame, f"STAFF #{track_id} {score:.2f}", (left, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            timestamp = round(frame_number / fps, 3)
            if staff is not None:
                track_id, score, left, top, right, bottom = staff
                csv_writer.writerow([
                    frame_number,
                    timestamp,
                    1,
                    1,
                    track_id,
                    round((left + right) / 2, 1),
                    round((top + bottom) / 2, 1),
                    round(score, 3),
                ])
            else:
                csv_writer.writerow([frame_number, timestamp, 0, 0, "", "", "", ""])
            writer.write(frame)
            frame_number += 1

    capture.release()
    writer.release()
    print(f"Wrote {frame_number} frames to {output_video}")
    print(f"Wrote frame results to {output_csv}")


if __name__ == "__main__":
    main()
