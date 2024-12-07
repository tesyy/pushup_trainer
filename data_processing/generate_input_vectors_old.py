import sys
import os

# Add the project root (PushUp) directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import PushupPosture as pp

import cv2
import mediapipe as mp

# import PushupPosture as pp
import numpy as np
import os
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if __name__ == '__main__':
    directory = './data/pushup_processed'

    video_names = sorted(os.listdir(directory))
    print("Videos to process:", video_names)

    file = open("./data/input_vectors.csv", "w")

    for video_name in video_names:
        if not video_name.endswith('.mp4'):
            print(f"Skipping non-video file: {video_name}")
            continue

        cap = cv2.VideoCapture("./data/pushup_processed/" + video_name)
        if not cap.isOpened():
            print(f"Failed to open video: {video_name}")
            continue

        frame_number = 0
        print(f"Processing video: {video_name}")

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print(f"End of video or empty frame in: {video_name}")
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                params = pp.get_params(results)
                if params is None or len(params) < 2:
                    print(f"Skipped frame {frame_number} in {video_name}: Invalid or insufficient params: {params}")
                    frame_number += 1
                    continue

                file.write("{},{},{},{}\n".format(
                    video_name[:3],
                    frame_number + 1,
                    params[0],
                    params[1]
                ))
                print(f"Writing to CSV: Video {video_name[:3]}, Frame {frame_number + 1}, Params: {params}")
                file.flush()
                frame_number += 1

        cap.release()
        print(f"Finished processing video: {video_name}")

    file.close()
    print("Processing complete. Results saved to ./data/input_vectors.csv.")

    cv2.destroyAllWindows()
