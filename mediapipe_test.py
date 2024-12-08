import cv2
import mediapipe as mp
import PushupPosture_ver1 as pp
import numpy as np
from utils import *
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (720, 1000))
# For video input:
cap = cv2.VideoCapture("data/correct.mp4")
#cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model("working_model_1.keras")

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            #continue
            break
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Resize the frame
        image_hight, image_width, _ = image.shape
        image_hight = 720
        image_width = 1000
        image = cv2.resize(image, (image_width, image_hight))
        params = pp.get_params(results)
        #flat_params = np.reshape(params, (1, 6))
        #output = model.predict(params)
        print(params)
        left_angle = params[0] if len(params) > 1 else 0
        right_angle = params[1] if len(params) > 2 else 0
        left_elbow_corx = int(params[2]*image_width) if left_angle > 0 else 0
        left_elbow_cory = int(params[3]*image_hight) if right_angle > 0 else 0
        right_elbow_corx = int(params[4]*image_width) if left_angle > 0 else 0
        right_elbow_cory = int(params[5]*image_hight) if right_angle > 0 else 0
        elbow_angle = int((left_angle+right_angle)/2)
        nomalized = 600/180 + 0.5
        color, word = ((0, 255, 0), "Deep Enough") if elbow_angle < 80 and elbow_angle > 0 else ((0, 0, 255), "Not Deep Enough")

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        coords = landmarks_list_to_array(results.pose_landmarks, image.shape)
        # label_params(image, params, coords)
        #break
        if right_angle > 0 and left_angle > 0:
            cv2.putText(
            image, str(int(left_angle)),
            (left_elbow_corx,left_elbow_cory),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1)
            cv2.putText(
            image, str(int(right_angle)),
            (right_elbow_corx,right_elbow_cory),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1)
        cv2.rectangle(image,
        (10, 700), (100, 200),
        color,
        2)
        if left_angle and right_angle > 0:
            cv2.rectangle(image,
            (10, 700),(100, int(100+elbow_angle*nomalized)),
            color,
            -1)
        else:
            cv2.rectangle(image,
            (10, 700),(100, 680),
            color,
            -1)
        cv2.putText(image, word,(5,160),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        #out.write(image)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
#out.release()
cap.release()
cv2.destroyAllWindows()
"""        ##
cv2.rectangle(image,
(0, 0),(image_hight-300, 74),
color,
-1)
cv2.putText(
image, "Back Is Sagging! :(",
(5,50),
cv2.FONT_HERSHEY_SIMPLEX,
1.5,
(0,0,0),
2)
##"""