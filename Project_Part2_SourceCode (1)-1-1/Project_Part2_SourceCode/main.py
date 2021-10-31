# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor

import os
cwd = os.getcwd()
print(cwd)
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

#obj = HandShapeFeatureExtractor.get_instance()
#image = "C:\\Users\\naikt\\OneDrive\\Documents\\MS\\Books\\CSE535_MC\\Midterm\\Project_Part2_SourceCode (1)-1-1\\Project_Part2_SourceCode\\00002.png"
#img = cv2.imread(image)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#result = obj.extract_feature(img)
#print(result)

#videopath = "C:\\Users\\naikt\\OneDrive\\Documents\\MS\\Books\\CSE535_MC\\Midterm\\Project_Part2_SourceCode (1)-1-1\\Project_Part2_SourceCode\\FanDown_PRACTICE_1_Naik.mp4"
#frames_path = "C:\\Users\\naikt\\OneDrive\\Documents\\MS\Books\\CSE535_MC\\Midterm\\Project_Part2_SourceCode (1)-1-1\\Project_Part2_SourceCode"
#frameExtractor(videopath,frames_path, 1)
#frameExtractor(videopath,frames_path, 1)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
listOfVideoNames = ["FanDown", "FanOff", "FanOn", "FanUp","LightOff","LightOn", "SetThermo","Num0", "Num1", "Num2", "Num3","Num4","Num5","Num6","Num7","Num8","Num9"]
#path = "C:\\Users\\naikt\\OneDrive\\Documents\\MS\\Books\\CSE535_MC\\Midterm\\Project_Part2_SourceCode (1)-1-1\\Project_Part2_SourceCode"
path = cwd
pathToVideos = path + "\\traindata\\Final_Videos_Naik_1222309959"
pathToImageFeatures = path + "\\traindata\\Feature_Images\\"

for i in range(1,18):
    print(pathToImageFeatures)
    #FanDown_PRACTICE_3_Naik
    for j in range(1,4):
        fileName = listOfVideoNames[i-1]+"_PRACTICE_"+str(j)+"_Naik.mp4"
        finalPath = pathToVideos+"\\"+fileName
        #print(finalPath)
        frameExtractor(finalPath,pathToImageFeatures,1,fileName)

obj = HandShapeFeatureExtractor.get_instance()
vectorArrays = []
#FanDown_PRACTICE_1_Naik.mp400002
for i in range(1,18):
    for j in range(1,4):
        requiredImage = pathToImageFeatures + listOfVideoNames[i - 1] + "_PRACTICE_" +str(j)+"_Naik.mp400002.png"
        print(requiredImage)
        img = cv2.imread(requiredImage)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = obj.extract_feature(img)
        print(np.asarray(result))
        result = np.asarray(result)

        tempList = result.ravel()
        vectorArrays.append(tempList)
        #result = obj.extract_feature()

#"C:\\Users\\naikt\\OneDrive\\Documents\\MS\\Books\\CSE535_MC\\Midterm\\Project_Part2_SourceCode (1)-1-1\\Project_Part2_SourceCode"
#    C:\Users\naikt\OneDrive\Documents\MS\Books\CSE535_MC\Midterm\Project_Part2_SourceCode(
#        1) - 1 - 1\Project_Part2_SourceCode\test\test - 1 - 2
    # =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

listOfTestVideos = ["0","1","2","3","4","5","6","7","8","9","DecreaseFanSpeed","Fan1On","Fan2Off","IncreaseFanSpeed","LightOff","LightOn","SetThermo",]
testPath = path + "\\test\\test-1-2\\T1-H-0.mp4"
testPath1 = path + "\\test\\test-1-2\\"
print("Path Name: ",os.path.basename(testPath))
frameExtractor(testPath,testPath1,1,"T1-H-0.mp4")
img = cv2.imread(testPath1 + "\\T1-H-0.mp400002.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = obj.extract_feature(img)
print(np.asarray(result))
result = np.asarray(result)
tempList1 = result.ravel()
from numpy import dot
from numpy.linalg import norm
import sys

for i in range(0,len(listOfTestVideos)):
    testPath = path + "\\test\\test-1-2\\T1-H-"+listOfTestVideos[i]+".mp4"
    testPath1 = path + "\\test\\test-1-2\\"
    print("Path Name: ", os.path.basename(testPath))
    fileName = os.path.basename(testPath)
    frameExtractor(testPath, testPath1, 1, fileName)
    img = cv2.imread(testPath1 + "\\"+fileName+"00002.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = obj.extract_feature(img)
    print(np.asarray(result))
    result = np.asarray(result)
    maxCosineValue = -1*sys.maxsize + 0.0
    tempList1 = result.ravel()
    ind = 0
    for j in range(0,51):
        cos_sim = dot(vectorArrays[j], tempList1)/(norm(vectorArrays[j])*norm(tempList1))
        print(cos_sim)
        if cos_sim > maxCosineValue:
            maxCosineValue = cos_sim
            ind = j
    print("Max Cosine Value: ",maxCosineValue,"Class Name: ", listOfVideoNames[ind//3])
    #break
    #print(cos_sim," ",listOfVideoNames[i//3])

#if 4.72015e-05 > 0.95581406:
#    print("true")