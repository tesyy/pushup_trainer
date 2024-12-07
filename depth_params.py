import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_params(results, exercise='pushup'):
    if results.pose_landmarks is None:
        # Adjust the size based on the number of parameters
        return np.zeros((1, 5))
        # Compute angles and distances
    # Example for back angle calculation
    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    # Extract required landmarks
    landmarks = results.pose_landmarks.landmark

    # Get coordinates
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
    right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
    left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
    right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

    # Back angle (shoulders, hips, ankles)
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    ankle_midpoint = (left_ankle + right_ankle) / 2

    #back_angle = calculate_angle(
    #    shoulder_midpoint, hip_midpoint, ankle_midpoint)
    back_angle = 0
    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value] and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value] and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value] is not None:

        # Elbow angle
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        right_elbow_angle = calculate_angle(
            right_shoulder, right_elbow, right_wrist)

        # Average elbow angle
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        left_armShoulder_angle = calculate_angle(left_wrist,left_elbow,left_shoulder)
        right_armShoulder_angle = calculate_angle(right_wrist,right_elbow,right_shoulder)
        params = list([left_elbow_angle, elbow_angle,left_armShoulder_angle,right_armShoulder_angle,left_elbow,right_elbow])
        return params
    else:
        params = [0,0,0,0,0,0]
        return params

