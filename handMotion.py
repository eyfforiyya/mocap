import cv2
import mediapipe as mp

print ("Import success!")

# Mediapipe setup

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Camera setup

cap = cv2.VideoCapture(1)

while True:
    # Read frame from camera
    success, img =  cap.read() 

    if not success:
        print("Can't receive frame (stream end?). Exiting...")
        break

#------------------------------------------------------------------------------------------------------------------------------------------------>    

    # Convert img to rgb for mediapipe (from BGR so the model can correctly detect hands)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process image frame to detect hands
    results = hands.process(img_rgb)
    
    
    
    
    
    
    
    # Show the video feed
    cv2.imshow("Camera feed", img) 
    
#------------------------------------------------------------------------------------------------------------------------------------------------>    

    # End stream
    if cv2.waitKey(1) == ord("q"):
        break
    