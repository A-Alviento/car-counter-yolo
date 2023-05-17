# import necessary packages
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Open video file for processing
cap = cv2.VideoCapture("./Videos/cars.mp4")
# load the yolov8 detection model
model = YOLO("../weights/yolov8l.pt")

# List of class names that the model can predict
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load a mask image. This can be used to focus detection in a certain region of the image.
mask = cv2.imread("mask.png")

""" 
Initialize a SORT tracker. This will be used to track detected objects across frames.

max_age is the maximum number of consecutive frames an object is allowed to be marked as 
"missed" before the tracker concludes that the object has left the scene and deletes its track

min_hits is the minimum number of times an object needs to be detected before a new track is created for it

If the IoU of a new detection with an existing track is above this iou_threshold, then the new 
detection is considered to be the same object as the existing track. 
A higher iou_threshold value will result in fewer tracks being created, 
as it requires a greater overlap for a new detection to be considered a separate object.
"""
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define a line for counting objects that cross it.
limits = [400, 297, 673, 297]

# Initialize a list to store IDs of objects that have crossed the line.
totalCount = []

while True:
    # Read a new frame from the video.
    success, img = cap.read()

    # Apply the mask to the frame.
    imgRegion = cv2.bitwise_and(img, mask)

    # Load a background graphics image and overlay it on the frame.
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Run the YOLO model on the masked region of the image.
    # device="mps" is for apple silicon to utilise gpu; remove if not needed
    results = model(imgRegion, stream=True, device="mps")

    # Initialize an empty array to store bounding boxes of detected objects.
    detections = np.empty((0, 5))

    # Loop over the detected objects in the image.
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the bounding box coordinates.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get the confidence level of the detection.
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class of the detected object.
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the object is a vehicle and the confidence is high enough.
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Add the bounding box and confidence level to the detections array.
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with the new detections.
    resultsTracker = tracker.update(detections)

    # Draw the counting line on the image.
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Loop over the tracked objects.
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a rectangle around the tracked object and print its ID on the image.
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Find the center of the bounding box.
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw a circle at the center of the bounding box.
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # If the center of the bounding box is on the counting line...
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # ...and the object's ID is not already in the total count list...
            if totalCount.count(id) == 0:
                # ...then add the ID to the total count list and highlight the counting line.
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Print the total count of vehicles that have crossed the line on the image.
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Display the image.
    cv2.imshow("Image", img)

    # Wait for a key press before moving to the next frame.
    cv2.waitKey(1)

