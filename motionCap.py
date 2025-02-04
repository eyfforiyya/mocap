import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)

if cap is None or not cap.isOpened():
    print("Unable to open video source ")
    exit()

while True:
    # Capture frames
    ret, frame = cap.read()
    # if ret is false, stop
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Frame operations
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display feed
    cv.imshow("frame", gray)

    if cv.waitKey(1) == ord("q"):
        break

# Release capture
cap.release()
cv.destroyAllWindows()
