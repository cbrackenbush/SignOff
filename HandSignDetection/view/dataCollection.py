import cv2
from cvzone.HandTrackingModule import HandDetector
# Initialize the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# Start capturing video from the webcam
while True:
    success, img = cap.read()
    if not success:
        break
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y:y+h, x:x+w]
        cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()