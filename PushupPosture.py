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
        params = np.array([left_armShoulder_angle,right_armShoulder_angle,left_elbow[0],left_elbow[1],right_elbow[0],right_elbow[1]])
        return params
    else:
        params = np.array([0,0,0,0,0,0])
        return params

def get_input_params(results):

    if results.pose_landmarks is None:
        # Adjust the size based on the number of parameters
        return list([0,0,0,0])
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

    points = {}
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    points["NOSE"] = np.array([nose.x, nose.y]) if nose.visibility > 0.5 else np.array([0, 0])
    left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    points["LEFT_EYE"] = np.array([left_eye.x, left_eye.y]) if left_eye.visibility > 0.5 else np.array([0, 0])
    right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    points["RIGHT_EYE"] = np.array([right_eye.x, right_eye.y,]) if right_eye.visibility > 0.5 else np.array([0, 0])
    mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    points["MOUTH_LEFT"] = np.array([mouth_left.x, mouth_left.y]) if mouth_left.visibility > 0.5 else np.array([0, 0])
    mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    points["MOUTH_RIGHT"] = np.array([mouth_right.x, mouth_right.y]) if mouth_right.visibility > 0.5 else np.array([0, 0])
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    points["LEFT_SHOULDER"] = np.array([left_shoulder.x, left_shoulder.y]) if left_shoulder.visibility > 0.5 else np.array([0, 0])
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    points["RIGHT_SHOULDER"] = np.array([right_shoulder.x, right_shoulder.y]) if right_shoulder.visibility > 0.5 else np.array([0, 0])
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    points["LEFT_ELBOW"] = np.array([left_elbow.x, left_elbow.y]) if left_elbow.visibility > 0.5 else np.array([0, 0])
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    points["RIGHT_ELBOW"] = np.array([right_elbow.x, right_elbow.y]) if right_elbow.visibility > 0.5 else np.array([0, 0])
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    points["RIGHT_WRIST"] = np.array([right_wrist.x, right_wrist.y]) if right_wrist.visibility > 0.5 else np.array([0, 0])
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    points["LEFT_WRIST"] = np.array([left_wrist.x, left_wrist.y]) if left_wrist.visibility > 0.5 else np.array([0, 0])
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    points["LEFT_HIP"] = np.array([left_hip.x, left_hip.y]) if left_hip.visibility > 0.5 else np.array([0, 0])
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    points["RIGHT_HIP"] = np.array([right_hip.x, right_hip.y]) if right_hip.visibility > 0.5 else np.array([0, 0])
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    points["LEFT_KNEE"] = np.array([left_knee.x, left_knee.y]) if left_knee.visibility > 0.5 else np.array([0, 0])
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    points["RIGHT_KNEE"] = np.array([right_knee.x, right_knee.y]) if right_knee.visibility > 0.5 else np.array([0, 0])
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    points["LEFT_ANKLE"] = np.array([left_ankle.x, left_ankle.y]) if left_ankle.visibility > 0.5 else np.array([0, 0])
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    points["RIGHT_ANKLE"] = np.array([right_ankle.x, right_ankle.y]) if right_ankle.visibility > 0.5 else np.array([0, 0])

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2
    points["MID_ANKLE"] = (points["LEFT_ANKLE"] + points["RIGHT_ANKLE"]) / 2
    points["MID_KNEE"] = (points["LEFT_KNEE"] + points["RIGHT_KNEE"]) / 2

    points["EYES"] = (points["RIGHT_EYE"] + points["LEFT_EYE"]) / 2
    points["MOUTH"] = (points["MOUTH_LEFT"] + points["MOUTH_RIGHT"]) / 2
    points["HEAD"] = (points["MOUTH"] + points["EYES"] + points["NOSE"]) / 3

    if (left_ankle or right_ankle) and (left_shoulder or right_shoulder) and (left_hip or right_hip):
        angles_shoulder_hip = calculate_angle(points["MID_SHOULDER"],points["MID_HIP"],points["MID_ANKLE"])
    elif (left_knee or right_knee) and (left_shoulder or right_shoulder) and (left_hip or right_hip):
        angles_shoulder_hip = calculate_angle(points["MID_SHOULDER"],points["MID_HIP"],points["MID_KNEE"])
    else:
        angles_shoulder_hip = 0

    if (points["HEAD"][0] != 0 or points["HEAD"][1] != 0) and (left_shoulder or right_shoulder) and (left_hip or right_hip):
        angles_head_hip = calculate_angle(points["HEAD"],points["MID_SHOULDER"],points["MID_HIP"])
    elif (points["HEAD"][0] != 0 or points["HEAD"][1] != 0) and (left_shoulder or right_shoulder) and (left_knee or right_knee):
        angles_head_hip = calculate_angle(points["HEAD"],points["MID_SHOULDER"],points["MID_KNEE"])
    elif (points["HEAD"][0] != 0 or points["HEAD"][1] != 0) and (left_shoulder or right_shoulder) and (left_ankle or right_ankle):
        angles_head_hip = calculate_angle(points["HEAD"],points["MID_SHOULDER"],points["MID_ANKLE"])
    else:
        angles_head_hip = 0

    if  left_wrist and (left_shoulder or right_shoulder):
        angles_shoulder_leftwrist = calculate_angle(points["MID_SHOULDER"],points["LEFT_SHOULDER"],points["LEFT_WRIST"])
    else:
        angles_shoulder_leftwrist = 0

    if  right_wrist and (left_shoulder or right_shoulder):
        angles_shoulder_rightwrist = calculate_angle(points["MID_SHOULDER"],points["RIGHT_SHOULDER"],points["RIGHT_WRIST"])
    else:
        angles_shoulder_rightwrist = 0
    params = list([angles_shoulder_hip,angles_head_hip,angles_shoulder_leftwrist,angles_shoulder_rightwrist])
    return params
