import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

image = cv2.imread("plat2.jpg")
#image = cv2.imread("plat2.jpg")
image = imutils.resize(image, width=300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255-gray
gray = cv2.GaussianBlur(gray,(5,5),0)
gray = cv2.bilateralFilter(gray, 11, 17, 17) 

cv2.imshow("gray", gray)
edged = cv2.Canny(gray, 30, 200) 
cv2.imshow("edged image", edged)

cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
cv2.imshow("Top 30 contours",image2)

i=7
for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
                screenCnt = approx
        x,y,w,h = cv2.boundingRect(c) 
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite('./'+str(i)+'.png',new_img)
        i+=1
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
cv2.waitKey(0)