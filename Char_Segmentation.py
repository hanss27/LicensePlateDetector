import cv2
import numpy as np
import functools
from pathlib import Path
import os

def detect_chars(imag, img_name,showSteps=False):
        image = cv2.imread(imag)
        image = cv2.resize(image,(300,120) )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = 255-gray
        gray = cv2.bilateralFilter(gray, 11, 17, 17) 
        thresh = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 43, 9)
        thresh = cv2.erode(thresh, (4,4))
        thresh = cv2.dilate(thresh, (3,3))
        thresh = cv2.dilate(thresh, (3,3))
        thresh = cv2.erode(thresh, (3,3), iterations=5)
        thresh = cv2.dilate(thresh, (3,3))

        _, labels = cv2.connectedComponents(thresh)


        mask = np.zeros(thresh.shape, dtype="uint8")

        # Set lower bound and upper bound criteria for characters
        total_pixels = image.shape[0] * image.shape[1]
        lower = total_pixels // 350 # heuristic param, can be fine tuned if necessary
        upper = total_pixels // 4 # heuristic param, can be fine tuned if necessary
        # Loop over the unique components
        for (i, label) in enumerate(np.unique(labels)):
                # If this is the background label, ignore it
                if label == 0:
                        continue
        
                # Otherwise, construct the label mask to display only connected component
                # for the current label
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
        
                # If the number of pixels in the component is between lower bound and upper bound, 
                # add it to our mask
                if numPixels > lower and numPixels < upper:
                        mask = cv2.add(mask, labelMask)
        if showSteps == True:
                cv2.imshow("thresh", thresh)
                cv2.imshow("gray", gray)
                cv2.imshow("mask",mask)

        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        # # Sort the bounding boxes from left to right, top to bottom
        # # sort by Y first, and then sort by X if Ys are similar
        def compare(rect1, rect2):
                if abs(rect1[1] - rect2[1]) > 10:
                        return rect1[1] - rect2[1]
                else:
                        return rect1[0] - rect2[0]

        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
        detected_char_list = []
        new_mask = mask.copy()
        for c in boundingBoxes:
                x,y,w,h = c
                x = x-2
                y = y-2
                w = w+2
                h = h+2
                if w>5 and h>10 and x > 5 and x < 290 and y > 5 and y < 60 : 
                        cv2.rectangle(new_mask,(x,y),(x+w,y+h),(0,0,0),3)
                        detected_char_list.append(mask[y:y+h, x:x+w])
        #cv2.imshow("detected_char",mask)
        output(detected_char_list, img_name)  # Take all detected_chars into a folder
        return detected_char_list

def output(detected_char_list, img_name):
        i = 1
        path = os.path.join(Path().absolute(), "tested") # Directory of current working directory, not __file__ 
        path = os.path.join(path, img_name[:-3])
        if not os.path.exists(path):
                os.makedirs(path)

        for pic in detected_char_list:  
                cv2.resize(pic, (70,100))
                #print(pic)
                pic_format = "test"+str(i)+".jpg" 
                cv2.imwrite(os.path.join(path, pic_format), pic)
                i+=1

def main():
        folder_path = "test" # Change the path with the folder with imgs
        path = os.path.join(Path().absolute(),folder_path) # Directory of current working directory, not __file__ 
        # img = "plat4.jpg" # Change it for a single testing
        # detected_char_list = detect_chars(os.path.join(path, img), img, showSteps=False)
       
       # Use this code below for loop all the imgs inside a folder
        path_list = os.listdir(path)
        for img in path_list:
                try:
                         mask = detect_chars(os.path.join(path,img), img, showSteps=False)
                except:
                        continue

        cv2.waitKey(0)
if __name__=="__main__":
        main()
