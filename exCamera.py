# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# Used object tracking code from the above attached link in combination with
# contour and rotated rectangle border code from the below attached link.
# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
 
while(1):
 
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
 
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    
    ret,thresh = cv.threshold(mask,127,255,0)
    contours,hierarchy = cv.findContours(thresh, 1, 2)
 

    if contours:
        cnt = max(contours, key=cv.contourArea)

        if len(cnt) >= 5:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
 
    cv.imshow('Frame',frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
 
cv.destroyAllWindows()

 
