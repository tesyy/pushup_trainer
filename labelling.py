import cv2
import os
import pandas as pd

# Path to the directory containing the processed pushup videos
# video_directory = 'data/pushup_processed/'
video_directory = 'data/test_processed/'

# List all video files in the directory
video_names = sorted(os.listdir(video_directory))

# Filter for MP4 files only
video_names = [v for v in video_names if v.endswith('.mp4')]

# Define label mapping based on video name (you may have to adjust this based on your dataset naming convention)
label_mapping = {
    "correct_form": "c",  # correct form
    "elbows_wide": "e",   # elbows wide
    "head_drop": "h",     # head drop
    "sagging_back": "b"    # sagging back
}

# Check for missing videos
# Generating expected video filenames
expected_videos = [f"{i:03}_correct_form.mp4" for i in range(14)] + \
                  [f"{i:03}_elbows_wide.mp4" for i in range(14)] + \
                  [f"{i:03}_head_drop.mp4" for i in range(14)] + \
                  [f"{i:03}_sagging_back.mp4" for i in range(14)]

print(expected_videos)

missing_videos = set(expected_videos) - set(video_names)
if missing_videos:
    print(f"Warning: Missing videos: {missing_videos}")

# Create a list to store the annotations
annotations = []

# Dictionary to store frame counts for each label
label_frame_counts = {label: 0 for label in label_mapping.values()}
label_video_counts = {label: 0 for label in label_mapping.values()}

# Loop over each video and its corresponding label
for video_name in video_names:
    # Extract the label from the video name
    for form_name, label in label_mapping.items():
        if form_name in video_name:
            break  # Break as soon as a match is found

    # Update video count for this label
    label_video_counts[label] += 1

    # Video path
    video_path = os.path.join(video_directory, video_name)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video FPS and total frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label_frame_counts[label] += total_frames

    print(f"Processing video: {video_name}, Label: {label}, FPS: {fps}, Total Frames: {total_frames}")

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate timestamp in seconds
        timestamp = frame_num / fps if frame_num < total_frames - 1 else "end"

        # Append annotation
        annotations.append({
            "video_name": video_name[:3],  # First 3 characters of the video name
            "timestamp": timestamp,
            "label": label
        })

        frame_num += 1

    cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()

# Convert the annotations list to a pandas DataFrame
annotations_df = pd.DataFrame(annotations)

# Save to CSV file
output_file = 'data/test_labels.csv'
annotations_df.to_csv(output_file, sep=',', index=False)

print(f"Annotations saved to {output_file}.")

# Dataset Composition Summary
total_videos = sum(label_video_counts.values())
total_frames = sum(label_frame_counts.values())

print("\nDataset Composition:")
for label, video_count in label_video_counts.items():
    frame_count = label_frame_counts[label]
    frame_percentage = (frame_count / total_frames) * 100

    print(f"{label}: {video_count} videos, {frame_count} total frames ({frame_percentage:.2f}%)")
