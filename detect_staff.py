"""Detect and track the staff member in a 3D-sensor video.

The detector finds people with YOLO and uses appearance similarity against
reference snapshots to identify the staff member. It produces an annotated
video and one CSV row per input frame.
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
        default=0.42,
        help="Appearance similarity threshold from 0 to 1.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Consecutive matching frames required before staff_present is true.",
    )
    return parser.parse_args()


def default_references() -> list[Path]:
    reference_dir = Path("sample/reference_images")
    return sorted(reference_dir.glob("page_1_image_[23].jpg"))


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
    output_avi = args.output_dir / "annotated.avi"
    output_csv = args.output_dir / "staff_frames.csv"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    compatible_writer = cv2.VideoWriter(
        str(output_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened() or not compatible_writer.isOpened():
        raise RuntimeError("Could not create output video writers")
    model = YOLO(args.model)
    stable_count = 0

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_number", "timestamp_seconds", "staff_present", "x", "y", "score"])

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

            best = None
            if result.boxes is not None:
                for box in result.boxes.xyxy.int().cpu().tolist():
                    left, top, right, bottom = box
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(width, right), min(height, bottom)
                    if right <= left or bottom <= top:
                        continue
                    crop = frame[top:bottom, left:right]
                    score = max(appearance_score(cv2, crop, ref) for ref in references)
                    if best is None or score > best[0]:
                        best = (score, left, top, right, bottom)

            is_match = best is not None and best[0] >= args.match_threshold
            stable_count = stable_count + 1 if is_match else 0
            staff_present = stable_count >= args.stable_frames
            x = y = ""
            if staff_present and best is not None:
                _, left, top, right, bottom = best
                x, y = round((left + right) / 2, 1), round((top + bottom) / 2, 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
                cv2.putText(frame, f"STAFF {best[0]:.2f}", (left, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            csv_writer.writerow([frame_number, round(frame_number / fps, 3), int(staff_present), x, y, round(best[0], 3) if best else ""])
            writer.write(frame)
            compatible_writer.write(frame)
            frame_number += 1

    capture.release()
    writer.release()
    compatible_writer.release()
    print(f"Wrote {frame_number} frames to {output_video}")
    print(f"Wrote compatibility video to {output_avi}")
    print(f"Wrote frame results to {output_csv}")


if __name__ == "__main__":
    main()
