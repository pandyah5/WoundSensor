import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(true):

    _, frame = cap.read()

    # Convert BGR to HSV color scheme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of wound red color in HSV

    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    
    # Threshold the HSV image to get only red colors that match with wound colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Scan the wound in the frame: ',(0,50), font, 1, (99,74,154), 3, cv2.LINE_AA)
    
    # Calculating percentage area
    try: 
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

        areacnt = cv2.contourArea(cnt)
        arearatio=((areacnt)/208154)*100
        
        boxes = []
        for c in cnt:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])

        boxes = np.asarray(boxes)
        # need an extra "min/max" for contours outside the frame
        left = np.min(boxes[:,0])
        top = np.min(boxes[:,1])
        right = np.max(boxes[:,2])
        bottom = np.max(boxes[:,3])
        
        cv2.rectangle(frame, (left,top), (right,bottom), (255, 0, 0), 2)
        
    except:
        pass
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    m = cv2.waitKey(5) & 0xFF
    k = cv2.waitKey(5) & 0xFF
    
    if k == 27: # Exit condition
        break
    
    # Press SpaceBar for 
    if m==32:
        print("The area of the wound is: ", arearatio * 0.6615, "cm squared.")
        print("The area of the Custom-Aid is: ", (right - left)*(bottom - top)*2.989*pow(10, -4), "cm squared.")
        print("The length is equal to: ", (right - left) / 95.23, "cm.")
        print("The width is equal to: ", (bottom - top) / 95.23, "cm.")
        
cv2.destroyAllWindows()
