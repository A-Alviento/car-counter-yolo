# import the necessary packages
from ultralytics import YOLO
import cv2

# downloads the yolov8l (large) detection model into current directory
# then it loads it into the model variable
# see https://docs.ultralytics.com/models/yolov8/#overview for more details
# model = YOLO('yolov8l.pt')

# once downloaded, move the downloaded model pt file into a folder called "weights"
# then use this code to run instead of the above 
model = YOLO('../weights/yolov8l.pt')

# run detection on an image
# show=True will show the image with bounding boxes
# device="mps" is for apple silicon to utilise gpu; remove if not needed
results = model("Images/3.png", show=True, device="mps")

# ensure image stays open until a key is pressed
cv2.waitKey(0)