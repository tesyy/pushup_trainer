import cv2
import mediapipe as mp
import numpy as np
import PushupPosture_ver1 as pp  
#import depth_params as dp # get_params for depth 
from tensorflow.keras.models import load_model
from utils import *  

# Load MediaPipe Pose components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load trained model
model = load_model("working_model_1.keras")


class VideoCamera:
    """
    VideoCamera class to access webcam.
    """
    def __init__(self):
        # Open the default webcam (0: default camera)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the camera when done
        self.video.release()


def run_pushup_posture_detection():
    """
    Function to perform push-up posture detection and display the live video feed with results.
    """
    camera = VideoCamera()  
    # cap = camera.video
    #cap = cv2.VideoCapture("data/test_processed/001_elbows_wide.mp4")
    cap = cv2.VideoCapture("data/correct.mp4")
    output_name = ['c', 'e', 'b', 'h']  # Labels: correct, elbows, back, head

    print("Starting Push-up Posture Detection...")
    print("Press 'Esc' to exit the program.")

    # Start MediaPipe Pose detection
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, image = cap.read()  # Read frame from camera
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Flip the image for a mirrored view
            image = cv2.flip(image, 1)

            # Get image dimensions
            image_hight, image_width, _ = image.shape

            # Convert BGR to RGB (required by MediaPipe) and resize
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hight = 720
            image_width = 1000
            image = cv2.resize(image, (image_width, image_hight))

            # Process the image to detect pose landmarks
            results = pose.process(rgb_image)

            # Extract posture parameters using get_params
            params = pp.get_input_params(results)
            
            # Reshape parameters to match model input shape: (1, 1, n_features)
            flat_params = np.reshape(params, (1, 1, 4))

            # Get model predictions
            output = model.predict(flat_params, verbose=0)
            # Adjust output probabilities
            output[0][0] *= 4  # Correct Form
            output[0][1] *= 0.7  # Elbows too wide
            output[0][2] *= 1.2    # Back sagging
            output[0][3] *= 2    # Head dropping
            max_index = np.argmax(output)
            output = np.zeros_like(output) 
            output[0, max_index] = 1
            print(output)

            # Normalize the outputs
            output = output / np.sum(output)

            # Determine the label based on probabilities
            label = ""
            for i in range(1, 4):  # Check elbows (e), back (b), head (h)
                if output[0][i] > 0.5:  # Threshold
                    label += output_name[i]

            # Default to "Correct Form" if no issues are detected
            if not label:
                label = "c"

            # Before calling label_final_results
            label = "".join([char for char in label if char in ['c', 'e', 'b', 'h']])  # Sanitize labels
            print(f"Predicted label: '{label}'")  # Debugging output

            depth_params = pp.get_params(results)
            left_angle = depth_params[0] if len(params) > 1 else 0
            right_angle = depth_params[1] if len(params) > 2 else 0
            left_elbow_corx = int(depth_params[2]*image_width) if left_angle > 0 else 0
            left_elbow_cory = int(depth_params[3]*image_hight) if right_angle > 0 else 0
            right_elbow_corx = int(depth_params[4]*image_width) if left_angle > 0 else 0
            right_elbow_cory = int(depth_params[5]*image_hight) if right_angle > 0 else 0
            elbow_angle = int((left_angle+right_angle)/2)
            nomalized = 600/180 + 0.5
            color, word = ((0, 255, 0), "Deep Enough") if elbow_angle < 80 and elbow_angle > 0 else ((0, 0, 255), "Not Deep Enough")

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
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

            # Display result
            label_final_results(image, label)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Push-up Posture Detection", image)

            # Break on 'Esc' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Exiting...")
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pushup_posture_detection()