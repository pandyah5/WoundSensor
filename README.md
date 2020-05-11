# WoundSensor
This project uses OpenCV library in python to detect fresh wounds and provides the dimension of bandages that will perfectly cover the wound.

# Problem Statement
Measuring the size of the wound is currently done manually which is highly inaccurate. As of now, many hospitals manually measure the size of wound to cut out band-aids that perfectly cover the wound. This makes it difficult, error-prone and time-consuming to monitor the healing process.

# Scope
To develop a program to get the dimensions of a customisable bandage that leads to a quicker and a more efficient healing process by speeding up the process of measuring the area of the wound, hence preventing any potential infections due to delay.

# Functionality of the program
1) Detects the wound using computer imaging algorithms and contour processing techniques.
2) Calculates the area covered by the wound.
3) Provides the optimum dimensions of a rectangular bandage that perfectly covers the patient’s wound.

# Algorithm used
1) The program selectively identifies the wound using a special range of HSV values.
2) The image is then converted into a contour image (Black & White) to compute its area.
3) The position of the extreme coordinates of the contour determines the corners of the rectangle.
4) The rectangular section is then displayed and its area is computed.

# Set up
The following libraries need to be installed prior to the exectution of the "main_script.py":
1) OpenCV [If 'import cv2' works fine then you are good to go]
   This link might help you: https://stackoverflow.com/questions/51853018/how-do-i-install-opencv-using-pip
2) pyttsx3 [If you want to activate voice commands, however this is optional]
