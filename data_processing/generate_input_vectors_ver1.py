import cv2
import mediapipe as mp
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import PushupPosture_ver1 as pp
import numpy as np
import os
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if __name__ == '__main__':
    #directory = './data/pushup_processed'
    directory = './data/test_processed'
    video_names = sorted(os.listdir(directory))

    # videos_to_use = ["000","001","002","003","004","005","006","007","008","009","010","023"]
    # video_names = [video + "_squat.mp4" for video in videos_to_use]

    #file = open("./data/New_input_vectors.csv", "w")
    file = open("./data/test_input_vectors.csv", "w")
    for video_name in video_names:
        cap = cv2.VideoCapture("./data/test_processed/" + video_name)
        frame_number = 0
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    # continue
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                image_hight, image_width, _ = image.shape

                params = pp.get_input_params(results)
                # print(params.shape)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                #label_params(image, params, coords)
                #print(frame_number)
                file.write("{},{},{},{},{},{}\n".format(
                    video_name[0:3],
                    frame_number+1,
                    params[0],
                    params[1],
                    params[2],
                    params[3]
                ))
                file.flush()
                frame_number += 1

                cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            print(video_name)

    file.close()

    cv2.destroyAllWindows()
