import cv2
import numpy as np

prev_frame = None
cap = cv2.VideoCapture(0)

if cap is None or not cap.isOpened():
    print("Unable to open video source ")
    exit()


while True:

    # Read the frames (two)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Trail layer (black layer where trail is drawn)
    trail = np.zeros_like(frame1, dtype=np.uint8)

    # absolute diff between frame 1 and 2
    frame_diff = cv2.absdiff(frame1, frame2)
    
    #grayscale
    gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # turns frame to b&w to contrast differences
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # fixes the spaces between thresholds to unify shapes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # detects the moving objects as white areas in the frame
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Quit
    if cv2.waitKey(1) == ord("q"):
        break


# Release capture
cap.release()
cv2.destroyAllWindows()
