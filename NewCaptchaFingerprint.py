# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 09:37:25 2016

@author: eshikasaxena
"""

import cv2
import numpy as np
import imutils
import math
import random
import uuid
import os

# Extract contours for fingerprint 
def getFingerprint(img):
    img = imutils.resize(img, width = 400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    skin = cv2.bitwise_and(gray, gray, mask = thresh) 
    clahe = cv2.createCLAHE(clipLimit = 40.0, tileGridSize = (8,8))
    cll = clahe.apply(skin)
    fingerprint = cv2.adaptiveThreshold(cll, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 15, C = 1)
    return gray, fingerprint    


fingersFound = 5
startCounting = False
captchaPassed = False
pts =[]

cap = cv2.VideoCapture(1) # start with rear camera if present 
if not cap.isOpened() :
    cap = cv2.VideoCapture(0)

    
# Generate random sequence for showing 1 to 5 fingers
captcha = range(1, 6)
random.shuffle(captcha)
capIndex = 0
#print captcha
    
while(cap.isOpened()):
    ret, img = cap.read()
    
    if startCounting :       
        caption = "Show " + str(captcha[capIndex]) + " fingers" #  + str(capIndex)
        cv2.putText(img, caption, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)     
    cv2.rectangle(img,(500,500),(100,100),(0,255,0),0)
    crop_img = img[100:500, 100:500]  

# Blur and threshold to extract contours
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (15, 15)   # (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    threshVal = 110    # 127
    # Use black background
#    unused, thresh1 = cv2.threshold(blurred, threshVal, 255,
#                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Use White background
    unused, thresh1 = cv2.threshold(blurred, threshVal, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_NONE
    max_area = -1
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]    #max contour for outline of hand

# draw the contour of the hand        
    x,y,w,h = cv2.boundingRect(cnt)
    hull0 = cv2.convexHull(cnt)   
    drawing = np.zeros(crop_img.shape,np.uint8)
    if startCounting :    
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0) 
        cv2.drawContours(drawing,[cnt],0,(0,255,0),0)  #draw hand
        cv2.drawContours(drawing,[hull0],0,(0,0,255),0) #draw hull
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0

#Get defect points to count fingers stretched
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90 :
            count_defects += 1
            if startCounting :
                cv2.circle(crop_img,far,10,[100,255,255],-1) # show gaps between fingers with yellow circles              
        if startCounting :  
            cv2.line(crop_img,start,end,[0,255,0],2)
            cv2.circle(crop_img,far,5,[0,0,255],-1)      # small red dots mark tracked points
      #      cv2.circle(crop_img,far,10,[100,255,255],3)  # yellow big circles


# Find center of the hand and plot tracking path
    if startCounting :  
        moments = cv2.moments(cnt)
        if moments['m00']!=0:            #Central mass of first order moments
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)
        cv2.circle(crop_img,centerMass,5,[255,0,100],2)   #Draw center mass
        cv2.circle(crop_img,centerMass,10,[255,255,255],3)
        font = cv2.FONT_HERSHEY_SIMPLEX   # label mid point cv2.putText(crop_img,'Mid',tuple(centerMass),font,1,(255,0,100),3) 
        
    	pts.append(centerMass)    # add to the points queue and loop over the tracked points
    	for i in xrange(1, len(pts)):
    		if pts[i - 1] is None or pts[i] is None:   # if tracked points are empty, ignore them
    			continue
    		cv2.line(crop_img, pts[i - 1], pts[i], (255, 0, 100), 5)  # draw connecting lines


# Count fingers being shown       
    if startCounting :      
        fingers = str(count_defects) + " fingers detected"
    elif captchaPassed == False:
        fingers = "Show 5 fingers to start"
    elif captchaPassed :
        fingers = "Get finger closeup & press <Tab>"
    cv2.putText(img,fingers, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)

    cv2.imshow('Gesture', img)
    if captchaPassed == False:    
        all_img = np.hstack((drawing, crop_img))
    elif captchaPassed == True:
        rawGrayImg, fingerprintImg = getFingerprint(img)
        all_img = np.hstack([rawGrayImg, fingerprintImg])
    cv2.imshow('Images', all_img)

    if (count_defects == fingersFound and captchaPassed == False) :
        startCounting = True
    
    if startCounting :    
        if captcha[capIndex] == count_defects:
#            print captcha[capIndex], count_defects  
            if capIndex < len(captcha)-1:          
                capIndex += 1
            else:
#                print "Completed Challenge"
                captchaPassed = True
                startCounting = False
                
    k = cv2.waitKey(10)    
    if k == 9 and captchaPassed :        # Press Tab key to take snap
        rawGrayImg, fingerprintImg = getFingerprint(img)        
#        cv2.imshow("Fingerprint saved", np.hstack([rawGrayImg, fingerprintImg]))
        cv2.imshow("Fingerprint saved", fingerprintImg)        
        if not os.path.exists("fingerprintDB") :
            os.makedirs("fingerprintDB")            
        temp = str(uuid.uuid4())
        fileName = ".\\fingerprintDB\\" + temp + ".jpg"    
        cv2.imwrite(fileName, fingerprintImg)

    elif k == ord('r') : #reset everything when "r" key pressed  NOT WORKING
        captchaPassed = False
        startCounting = False    
    elif k == 27:
        break       
             
    
cap.release()
cv2.destroyAllWindows()