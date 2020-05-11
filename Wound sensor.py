#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import math
from random import randrange
import time
import msvcrt as m
import pyttsx3

cap = cv2.VideoCapture(0)
points = 0
go_on = 0
while(1):
    cont = 0
    try: 
        #an error comes if it does not find anything in window as it cannot find contour of max area
        #therefore this try error statement
        
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)

        #define region of interest
        roi=frame[10:300, 10:300]


        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # define range of wound color in HSV
        lower_skin = np.array([356,70,64], dtype=np.uint8)
        upper_skin = np.array([352,55,43], dtype=np.uint8)
        
        #extract skin color image  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100)
        
        #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
        #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        
        #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame,'The area of the wound is: ',(0,50), font, 2, (0,0,235), 3, cv2.LINE_AA)
        
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[ ]:


## Use this link for color selector http://colorizer.org/


# In[ ]:


import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
     
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
        
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[200:600, 200:600]
        
        cv2.rectangle(frame,(200,200),(400,400),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([356,70,64], dtype=np.uint8)
        upper_skin = np.array([352,55,43], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        
        
    #find contours
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
        #print(arearatio)
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'The area of the wound is: ',(0,50), font, 1, (0,0,255), 3, cv2.LINE_AA)
        #cv2.putText(frame,arearatio,(0,100), font, 2, (0,0,255), 3, cv2.LINE_AA)

        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    


# ## Source code

# In[12]:


import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
     
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:600, 100:600]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,0,30], dtype=np.uint8)
        upper_skin = np.array([80,80,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
        #print("Here2")
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX

        #show the windows
        cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    


# In[1]:


#For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
#Different softwares use different scales. 
#So if you are comparing OpenCV values with them, you need to normalize these ranges.


# # THIS ONE WORKS 

# In[13]:


import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0,120,70])
    upper_blue = np.array([10,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Scan the wound in the frame: ',(0,50), font, 1, (99,74,154), 3, cv2.LINE_AA)
    ###
    # Calculating percentage area
    try: 
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)

        areacnt = cv2.contourArea(cnt)
        arearatio=((areacnt)/208154)*100
        ###
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
        ###
    except:
        pass
    
    
    ### 
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    m = cv2.waitKey(5) & 0xFF
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if m==32:
        print("The area of the wound is: ", arearatio * 0.6615, "cm squared.")
        print("The area of the band is: ", (right - left)*(bottom - top)*2.989*pow(10, -4), "cm squared.")

cv2.destroyAllWindows()


# In[4]:


## Code to find HSV colors
## Now you take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound respectively. 


# In[13]:


red = np.uint8([[[255,0,0 ]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print (hsv_red)


# In[ ]:




