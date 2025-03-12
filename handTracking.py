import cv2
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import landmark_pb2

print ("Import success!")

# Mediapipe setup

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Camera setup
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:

    while cap.isOpened():
        # Read frame from camera
        success, image =  cap.read() 

        if not success:
            print("Can't receive frame (stream end?). Exiting...")
            break #break is used when using video feed, continue is for still images

#------------------------------------------------------------------------------------------------------------------------------------------------>    

        # Convert image from feed from BGR to RGB to accurately track hands
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Convert image back to BGR so OpenCV can display it
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#------------------------------------------------------------------------------------------------------------------------------------------------>    

        # Draw the landmarks for each detected hand (default)
        
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             image,
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing_styles.get_default_hand_landmarks_style(),
        #             mp_drawing_styles.get_default_hand_connections_style()           
        #         )
        
#------------------------------------------------------------------------------------------------------------------------------------------------>    

        # # Display only the fingertips
        # fingertip_indices = [4, 8, 12, 16, 20]
        
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         #Subset of only fingertip landmarks positions
        #         fingertip_landmarks = landmark_pb2.NormalizedLandmarkList(
        #             landmark = [hand_landmarks.landmark[i] for i in fingertip_indices]
        #         )
                
        #     mp_drawing.draw_landmarks(
        #         image,
        #         fingertip_landmarks,
        #         connections=None
        #     )

#------------------------------------------------------------------------------------------------------------------------------------------------>    

        # Map fingertip landmarks to RGB values (index finger only)
        
        if results.multi_hand_landmarks:
            
            index_fingertips = [] # array to store each index fingertip
            
            for hand_landmarks in results.multi_hand_landmarks:
                index_fingertip = hand_landmarks.landmark[8]
                index_fingertips.append(index_fingertip)
                
                fingertip_landmarks = landmark_pb2.NormalizedLandmarkList(landmark = index_fingertips)

            mp_drawing.draw_landmarks(
                image,
                fingertip_landmarks,
                connections=None
            )
            
            # RGB Overlay
            tint = np.full_like(image, (50, 0, 0), dtype = np.uint8)
            blended = cv2.addWeighted(image, 0.5, tint, 0.5, 0)
            
            # https://chuoling.github.io/mediapipe/solutions/hands#python-solution-api
            # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
            
        else:
            blended = image

#------------------------------------------------------------------------------------------------------------------------------------------------>         
        
        # Show the video feed
        cv2.imshow("Hand tracking", cv2.flip(blended, 1)) 

#------------------------------------------------------------------------------------------------------------------------------------------------>    

        # End stream
        if cv2.waitKey(1) == ord("q"):
            break
        
cap.release()
    