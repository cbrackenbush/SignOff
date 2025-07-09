import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
# This script captures video from the webcam, detects hands, and crops the detected hand region.

# Initialize the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20  # Offset for cropping the image
# Function to ensure the bounding box is within image bounds

imgSize = 300  # Size to which the cropped image will be resized
# Start capturing video from the webcam

folder = "Data/Z"

counter = 0  # Counter for saving images

while True:
    success, img = cap.read()
    if not success:
        break
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3) , np.uint8)*255  # Create a white background for the cropped image
        # Ensure coordinates are within image bounds
        imgCrop = img[y-offset:y+ h+offset, x-offset:x+ w+offset]
        # Check that the cropped image has valid size
        imgCropShape = imgCrop.shape


        apsectRatio = h/w

        if apsectRatio > 1:
            k = imgSize / h 
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wGap+wCal] = imgResize  # Place the cropped image on the white background

        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap+hCal, :] = imgResize

        if imgCrop.size > 0:
            # Ensure the cropped image is resized to a fixed size
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) 
    if key == ord('s'):
        counter += 1  
        # Save the cropped image to the specified folder
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)