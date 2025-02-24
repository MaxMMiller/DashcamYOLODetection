import cv2
from ultralytics import YOLO
import numpy as np
##import torch ##loads yolo model
##model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = YOLO("yolo11s.pt")

# Access the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")




while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLO
    results = model(frame)

    # Render the results on the frame
    rendered_frame = results.render()[0]

    # Display the resulting frame
    cv2.imshow('YOLO Webcam', rendered_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows