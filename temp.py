import cv2

cap = cv2.VideoCapture(0)
_, im = cap.read()

print im