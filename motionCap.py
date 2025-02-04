import cv2
import numpy as np

prev_frame = None  # Initial frame
cap = cv2.VideoCapture(1)

if cap is None or not cap.isOpened():
    print("Unable to open video source ")
    exit()

while True:
    # Capture frames
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Frame operations
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # fmt: off
    frame_diff = cv2.absdiff(prev_frame, gray)  # Difference between previous frame and current
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY) # turns frame to b&w to contrast differences
    thresh = cv2.dilate(thresh, None, iterations=2) # fixes the spaces between thresholds to unify shapes
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # detects the moving objects as white areas in the frame

    if contours:
        largest_contour = max(contours, key = cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # fmt: on

    # Display feed
    # cv2.imshow("Gray frame", gray)
    # cv2.imshow("Difference Frame", frame_diff)
    # cv2.imshow("Threshold Frame", thresh)
    cv2.imshow("Motion detection", frame)

    # Update the prev_frame
    prev_frame = gray.copy()

    if cv2.waitKey(1) == ord("q"):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
