# import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# for webcam
cap = cv2.VideoCapture(0) 
# set width and height
cap.set(3, 1280)
cap.set(4, 720)

# for video file
#cap = cv2.VideoCapture("../Videos/motorbikes.mp4")

# load the yolov8 detection model
model = YOLO("../weights/yolov8l.pt")

# list of class names by index, i.e. class_names[0] == "person"
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

# Initialize variables to calculate frames per second (FPS)
prev_frame_time = 0
new_frame_time = 0

# loop through frames
while True:
    # Get the current time. This will be used to calculate FPS.
    new_frame_time = time.time()

    # read the frame from the webcam
    success, img = cap.read()

    # run detection on the frame; device="mps" is for apple silicon to utilise gpu; remove if not needed
    results = model(img, stream=True, device="mps")

    # loop over the detected objects in image
    for r in results:
        boxes = r.boxes

        # loop over the bounding boxes and draw them on the image
        for box in boxes:
            # get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # draw a rectangle around the detected object
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # get the confidence score for the detected object
            conf = math.ceil((box.conf[0] * 100)) / 100

            # get the class of the detected object
            cls = int(box.cls[0])

            # print the class and confidence score on the bounding box
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    
    # Calculate and print FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
