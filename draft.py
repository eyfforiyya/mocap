# Testing zone

import cv2
import numpy as np
from pynput.mouse import Controller

print("Import success!")

mouse = Controller() #this is the constructor, yayy! i understand it 

width, height = 1920, 1080
image = np.full((720, 1280, 3), (255, 255, 255), dtype=np.uint8)

while True:
    #get mouse position
    mouse_x, mouse_y = mouse.position
    
    #normalize
    mouse_x = mouse_x / width   
    mouse_y = mouse_y / height
    
    red = int(mouse_x * 255)
    green = int(mouse_y * 255)
    blue = 255 - red #test
    
    # create tint color 
    tint = np.full_like(image, (red, green, blue), dtype = np.uint8)
    blended = cv2.addWeighted(image, 0.5, tint, 0.5, 0)
    
    # Show the result
    cv2.imshow("Mouse-Controlled RGB Tint", blended)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()







