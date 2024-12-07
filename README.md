# PushUp Trainer

## Overview

**PushUp_Trainer** is a posture tracking and machine learning tool designed to monitor and analyze exercise form, specifically for push-ups. The system uses pose tracking techniques to evaluate your posture and provide corrective suggestions for improvement.

This project was developed as part of the course **COMP4471 Deep Learning in Computer Vision**.

## Features

- **Pose Tracking**: Uses computer vision and machine learning models to track body posture during push-ups.
- **Real-Time Feedback**: Provides feedback on posture and form during exercise.
- **Corrective Suggestions**: Analyzes posture and suggests corrective actions to improve form and prevent injuries.
- **Data-Driven Insights**: Tracks your progress over time and helps optimize your push-up performance.
- **Posture Issue Detection**: Identifies common form issues such as elbows too wide, head dropping, and back sagging, and provides suggestions for correction.

## Posture Identification

The system identifies specific posture issues during push-ups and assigns a code to each form issue:

- **‘c’**: Correct form — Your posture is perfect.
- **‘e’**: Elbows too wide — The elbows are flaring out too much, which can strain the shoulders.
- **‘h’**: Head dropping — The head is not aligned with the body, which can lead to neck strain.
- **‘b’**: Back sagging — The back is not in a straight line, which can cause lower back pain.

These identifiers help you know exactly where to focus for improving your form.

## How It Works

The system leverages advanced **pose estimation** techniques to analyze body movements during push-ups. It tracks key body landmarks, such as the shoulders, elbows, and hips, and evaluates their alignment. Based on this data, it provides suggestions to help you perform push-ups with proper form.

## Credits
This project is based on the work of: twixupmysleeve https://github.com/twixupmysleeve/Posture which helped form the basis of posture detection in this system.

