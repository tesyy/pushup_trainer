import cv2
import numpy as np


def landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape

    if landmark_list is None:
        return None

    return np.asarray([(lmk.x * cols, lmk.y * rows)
                       for lmk in landmark_list.landmark])


def label_params(frame, params, coords):

    if coords is None or params is None:
        return

    params = params * 180 / np.pi  # Convert radians to degrees

    # Label Back Angle
    shoulder_left = coords[11]
    shoulder_right = coords[12]
    shoulder_mid = (shoulder_left + shoulder_right) / 2
    cv2.putText(frame, f'Back Angle: {params[0]:.2f}', (int(shoulder_mid[0]), int(shoulder_mid[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Label Elbow Angle
    elbow_left = coords[13]
    elbow_right = coords[14]
    elbow_mid = (elbow_left + elbow_right) / 2
    cv2.putText(frame, f'Elbow Angle: {params[1]:.2f}', (int(elbow_mid[0]), int(elbow_mid[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def label_final_results(image, label):
    # Expanded labels dictionary
    expanded_labels = {
        "c": "Correct Form",
        "e": "Elbows are too wide, keep them closer",
        "b": "Back is sagging, keep your back straight",
        "h": "Head is dropping, keep your head up",
    }

    # Sanitize label to keep only valid keys
    valid_labels = set(expanded_labels.keys())
    label_list = [char for char in label if char in valid_labels]

    # Map labels to descriptions safely using get
    described_label = [expanded_labels.get(l, f"Unknown: {l}") for l in label_list]

    # Combine descriptions
    final_description = ", ".join(described_label)

    # Determine rectangle color: green for 'Correct Form', blue otherwise
    if "c" in label_list:
        color = (42, 210, 48)  # Green for 'Correct Form'
    else:
        color = (0, 0, 255)  # Red for 'Elbows too wide', 'Back sagging', or 'Head dropping'

    # Draw background rectangle
    image_height, _, _ = image.shape
    cv2.rectangle(image,
                  (0, 0), (image_height, 74),
                  color,
                  -1
                  )

    # Display text description on image
    cv2.putText(
        image, "   " + " + ".join(described_label),
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
