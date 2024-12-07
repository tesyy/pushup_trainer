import os
import csv
import cv2
import pandas as pd

def get_total_frames(video_name):
    # Construct the filename with the correct pattern
    filename = f"data/pushup_processed/{video_name}.mp4"  # Full name, e.g., "000_correct_form.mp4"
    print(f"Attempting to open: {filename}")
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return 0  # Return 0 if the file does not exist

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Failed to open video: {filename}")
        return 0  # Return 0 if the video cannot be opened
    
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames


# Read the labels and prepare video names
rows = []

fps = 12
form_mapping = {
    'c': 'correct_form',
    'e': 'elbows_wide',
    'h': 'head_drop',
    'b': 'sagging_back'
}

# Read labels from CSV file
with open('./data/labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    # Skip header (first row)
    next(csv_reader)

    for r in csv_reader:
        # r is a list, so we directly access each element
        video_number = r[0]  # The number (e.g., "000")
        timestamp = r[1]  # Timestamp in seconds (could be float or 'end')
        label = r[2]  # The label (e.g., "c", "e", "h", or "b")
        
        # Construct the full video name by appending the appropriate form (used for frame extraction)
        video_name = f"{video_number}_{form_mapping[label]}"  # e.g., "000_correct_form"
        
        # Add the full video name, timestamp, and label to the rows list
        rows.append((video_name, video_number, timestamp, label))  # Add video number as separate item
        line_count += 1
    print(f'Processed {line_count} lines.')

# Open the output file for writing
file = open("./data/output_vectors.csv", "w")
frame_number = 0

# Process each row (video info) from the labels file
for row in rows:
    full_video_name = row[0]  # Full video name including form (used for frame extraction)
    video_number = row[1]  # Video number (used for writing to output)

    # Get the total frames for the current video
    total_frames = get_total_frames(full_video_name)  # Get the total number of frames for the full video
    if total_frames == 0:
        continue  # Skip if there are issues with the video (e.g., file not found)
    
    if "end" in row[2]:  # Check if the timestamp is 'end'
        end_frame = int(total_frames)
    else:
        end_frame = int(float(row[2]) * fps)  # Convert timestamp to frame number based on fps

    # Write frame information for the current video
    for i in range(frame_number, end_frame):
        # Set binary values for the labels
        c = 1 if "c" in row[3] else 0
        b = 1 if "b" in row[3] else 0
        e = 1 if "e" in row[3] else 0
        h = 1 if "h" in row[3] else 0

        frame_number += 1

        # Write the data for each frame
        # Only use the video number (e.g., "000") for output
        line = "{},{},{},{},{},{}\n".format(video_number, frame_number, c, b, e, h)
        file.write(line)
    file.flush()

    if "end" in row[2]:  # Reset the frame number if 'end' is encountered
        frame_number = 0

file.close()

# After generating the output_vectors.csv, we filter it based on input_vectors.csv

# Read input_vectors.csv and output_vectors.csv
input_df = pd.read_csv('data/input_vectors.csv', header=None)
output_df = pd.read_csv('data/output_vectors.csv', header=None)

# Extract the first two columns (the matching criteria) from input
input_set = set(zip(input_df[0], input_df[1]))

# Filter output_df by checking if the first two columns are in input_set
filtered_output_df = output_df[output_df.apply(lambda row: (row[0], row[1]) in input_set, axis=1)]

# Ensure video_number column (first column) has leading zeros
filtered_output_df[0] = filtered_output_df[0].apply(lambda x: str(x).zfill(3))

# Overwrite the output_vectors.csv with the filtered content
filtered_output_df.to_csv('data/output_vectors.csv', index=False, header=False)

print("Filtered output saved to 'output_vectors.csv'")
