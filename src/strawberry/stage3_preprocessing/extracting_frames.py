import cv2
import os
from datetime import datetime, timedelta
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

cv2.setNumThreads(1)

def normalize_date_folder(folder_name):
    """
    Convert:
        2026-03-21 -> 21-03-2026
        21-03-2026 -> 21-03-2026

    Return None if folder name is not a valid date format.
    """

    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", folder_name):
            dt = datetime.strptime(folder_name, "%Y-%m-%d")
            return dt.strftime("%d-%m-%Y")

        if re.match(r"^\d{2}-\d{2}-\d{4}$", folder_name):
            dt = datetime.strptime(folder_name, "%d-%m-%Y")
            return dt.strftime("%d-%m-%Y")

    except ValueError:
        pass

    return None

PROJECT_ROOT = Path(__file__).resolve().parents[3]


data_root = PROJECT_ROOT / "data"
raw_root = data_root / "01_raw" / "strawberry"
processed_root = data_root / "02_processed" / "strawberry"


script_dir = os.path.dirname(os.path.abspath(__file__))

sample_minutes = 8
save_workers = max(1, min(8, os.cpu_count() or 1)) # use between 1 and 8 threads for saving frames, can adjust based on performance needs and available CPU cores, for example if your CPU has 4 cores, you can set to 4 or 8 for better performance, but if you have only 2 cores, setting to 8 may cause more overhead than benefit, so adjust accordingly
max_pending_writes = save_workers * 4 # limit the number of pending writes to avoid memory issues, can adjust based on available memory and performance needs


def save_frame(filename, frame):
    return cv2.imwrite(filename, frame)


def video_sort_key(filename):
    match = re.search(r"_(\d{2}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    return filename


def main():
    date_folders = []

    for item in os.listdir(raw_root):

        folder_path = raw_root / item

        if not folder_path.is_dir():
            continue

        normalized_date = normalize_date_folder(item)

        has_videos = any(
            path.is_file() and path.suffix.lower() == ".mp4"
            for path in folder_path.iterdir()
        )

        if normalized_date is not None and has_videos:
            date_folders.append((folder_path, normalized_date))

    if not date_folders:
        print(f"No valid raw video folders found in: {raw_root}")
        return
    
    date_folders.sort(
    key=lambda x: datetime.strptime(x[1], "%d-%m-%Y")
    )

    with ThreadPoolExecutor(max_workers=save_workers) as executor:
        for video_folder, output_date in date_folders:

            print(f"\n{'='*60}")
            print(f"Processing folder: {video_folder.name}")
            print(f"{'='*60}")

            output_folder = processed_root / f"frames_{output_date}"
            os.makedirs(output_folder, exist_ok=True)

            csv_path = output_folder / "all_frames.csv"

            video_files = sorted(
                [
                    f
                    for f in os.listdir(video_folder)
                    if f.lower().endswith(".mp4")
                ],
                key=video_sort_key
            )

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["video_name", "frame_number", "timestamp", "file_directory"])

                pending_writes = deque()
                frame_index = 1

                # loop through videos following the sorted order
                for video_file in video_files:

                    print(f"\nProcessing: {video_file}")

                    video_path = os.path.join(video_folder, video_file)

                    cap = cv2.VideoCapture(video_path)

                    # skip video if cannot open
                    if not cap.isOpened():
                        print("[SKIP] Cannot open video:", video_file)
                        continue

                    # collect start time from filename
                    try:
                        time_part = video_file.split("_")[1].replace(".mp4", "")
                        start_time = datetime.strptime(time_part, "%H-%M-%S")
                    except Exception:
                        print("[SKIP] Wrong filename:", video_file)
                        cap.release()
                        continue

                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if fps == 0:
                        print("[SKIP] FPS = 0:", video_file)
                        cap.release()
                        continue

                    interval = max(1, int(fps * 60 * sample_minutes)) # 1 frame every 8 minutes

                    frame_count = 0

                    # extract frames
                    while True:

                        if frame_count % interval == 0:
                            ret, frame = cap.read()
                        else:
                            ret = cap.grab()
                            frame = None

                        if not ret:
                            break

                        if frame_count % interval == 0:

                            current_time = start_time + timedelta(seconds=frame_count / fps)
                            timestamp = current_time.strftime("%H-%M-%S")

                            # not using frame number in filename to avoid confusion, instead using timestamp which is more meaningful, for example:
                            # frame-1_12-00-00.jpg
                            # frame-2_12-05-00.jpg
                            filename = os.path.join(
                                output_folder,
                                f"frame-{frame_index}_{timestamp}.jpg"
                            )

                            future = executor.submit(save_frame, filename, frame.copy())
                            row = [video_file, frame_index, timestamp, filename]
                            pending_writes.append((future, row))
                            frame_index += 1

                            if len(pending_writes) >= max_pending_writes:
                                done_future, done_row = pending_writes.popleft()
                                if done_future.result():
                                    writer.writerow(done_row)
                                else:
                                    print("[SKIP] Cannot save frame:", done_row[3])

                        frame_count += 1

                    cap.release()

                while pending_writes:
                    done_future, done_row = pending_writes.popleft()
                    if done_future.result():
                        writer.writerow(done_row)
                    else:
                        print("[SKIP] Cannot save frame:", done_row[3])

            print("\nDone extracting frames. CSV saved at:", csv_path)
    
if __name__ == "__main__":
    main()
