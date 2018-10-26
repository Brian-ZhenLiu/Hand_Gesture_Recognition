import numpy as np
from numpy import *
from PIL import Image
from skimage import feature
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import cv2
import os
import glob
import time
import sys
import pickle
import pandas as pd

def isInRange(x, y, W, H):
    if x >= W:
        return False
    elif x < 0:
        return False
    elif y >= H:
        return False
    elif y < 0:
        return False
    else:
        return True

def recursive(x, y, imgArr, num):
 (W, H) = imgArr.shape

 if isInRange(x - 1, y, W, H):
  if imgArr[y, x - 1] == 255:
   imgArr[y, x - 1] = num
   recursive(x - 1, y, imgArr, num)

 if isInRange(x - 1, y + 1, W, H):
  if imgArr[y + 1, x - 1] == 255:
   imgArr[y + 1, x - 1] = num
   recursive(x - 1, y + 1, imgArr, num)

 if isInRange(x, y + 1, W, H):
  if imgArr[y + 1, x] == 255:
   imgArr[y + 1, x] = num
   recursive(x, y + 1, imgArr, num)
    
 if isInRange(x + 1, y + 1, W, H):
  if imgArr[y + 1, x + 1] == 255:
   imgArr[y + 1, x + 1] = num
   recursive(x + 1, y + 1, imgArr, num)
    
 if isInRange(x + 1, y, W, H):
  if imgArr[y, x + 1] == 255:
   imgArr[y, x + 1] = num
   recursive(x + 1, y, imgArr, num)
    
 if isInRange(x + 1, y - 1, W, H):
  if imgArr[y - 1, x + 1] == 255:
   imgArr[y - 1, x + 1] = num
   recursive(x + 1, y - 1, imgArr, num)
    
 if isInRange(x, y - 1, W, H):
  if imgArr[y - 1, x] == 255:
   imgArr[y - 1, x] = num
   recursive(x, y - 1, imgArr, num)
    
 if isInRange(x - 1, y - 1, W, H):
  if imgArr[y - 1, x - 1] == 255:
   imgArr[y - 1, x - 1] = num
   recursive(x - 1, y - 1, imgArr, num)


def getFilePath(path):
 rgbImgPathList = []
 depImgPathList = []
 for filename in os.listdir(path):
  if filename == ".DS_Store":
   continue
  path_ = path + "/" + filename
  for filename_ in os.listdir(path_):
   if filename_ == ".DS_Store":
    continue
   for i in range(1, 11):
    rgbImgPathList.append(path_ + "/" + filename_ + "/" + str(i) + "_rgb.png")
    depImgPathList.append(path_ + "/" + filename_ + "/" + str(i) + "_depth.png")

 return rgbImgPathList, depImgPathList


def svcGridSearch(XTrains, XTests, YTrains, YTests, k):
 print("training model by using SVC .........")
 start_time = time.time()

 # k cross validation, one vs one, testing several cost penalty 
 svc = GridSearchCV(svm.SVC( decision_function_shape="ovo"),
        param_grid=[ { "kernel":["rbf", "linear", "poly"],
              "C":[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                              "gamma": [1/0.1**2, 1/0.5**2, 1, 1/2**2, 1/4**2],
                              "degree": [2, 3] } ],
        cv=k, n_jobs=-1).fit(XTrains, YTrains)
 end_time = time.time()
 print("training model took :", str(end_time - start_time), " sec")
 return svc

def SVC(X, Y, k):
 X = np.array(X)
 Y = np.array(Y).flatten()

 # seperate X and Y into training set and testing set
 # the ratio of training set to testing set is 7:3
 XTrains, XTests, YTrains, YTests = train_test_split(X, Y, test_size=0.2, random_state=1)

 # using grid search tool to test several parameters
 svc = svcGridSearch(XTrains, XTests, YTrains, YTests, k)
 filename = 'svc.sav'
 pickle.dump(svc, open(filename, 'wb'))
 print(confusion_matrix(YTests, svc.predict(XTests)))

 print("Kernel : " + str(svc.best_estimator_.kernel) + 
  "\nUsing 10-fold cross-validation" +
  ", mean score for cross-validation is " + str(svc.best_score_) +
  "\nPenalty parameter C = " + str(svc.best_estimator_.C) +
  "\nGamma = " + str(svc.best_estimator_.gamma) +
  "\nDegree = " + str(svc.best_estimator_.degree) +
  ", Accuracy score = " + str(metrics.accuracy_score(YTests, svc.predict(XTests))) )

def getHand(dep):
  handImg = np.zeros(dep.shape)
  handPixNum = 0
  count = 14
  while handPixNum < 7000:
   handPixs = where(dep == count)
   handImg[handPixs] = 255
   handPixNum = handPixNum + len(handPixs[0])
   count = count + 1
  return handImg

def denoise(handImg):
  handImg[where(handImg != 255)] = 0
  handPixs = where(handImg != 0)
  num = 0
  handArea = 0
  biggestNun = 0
  for i in range(len(handPixs[0])):
   (y, x) = handPixs[0][i], handPixs[1][i]
   if handImg[y][x] == 255:
    num = num + 1
    handImg[y][x] = num
    recursive(x, y, handImg, num)
    temp = len(where(handImg == num)[0])
    if temp > handArea:
     handArea = temp
     biggestNun = num

  for i in range(1, num + 1):
   if i == biggestNun:
    y_locArr, x_locArr = where(handImg == i)
    continue
   handImg[where(handImg == i)] = 0
  return handImg, y_locArr, x_locArr

def crop(handImg, y_locArr, x_locArr):
  x1 = x_locArr[argmin(x_locArr)]
  y1 = y_locArr[argmin(y_locArr)]
  x2 = x_locArr[argmax(x_locArr)]
  y2 = y_locArr[argmax(y_locArr)]
  return x1, x2, y1, y2




def generateDtaset(rgbImgPathList, depImgPathList, XAtt, yAtt):
 print("Generating dataset.....")
 while len(rgbImgPathList) != 0:
  rgbPath = rgbImgPathList.pop()
  print(rgbPath)
  img = cv2.imread(rgbPath)
  img = cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)

  depPath = depImgPathList.pop()
  dep = cv2.imread(depPath, 0)
  # get hand
  handImg = getHand(dep)
  # drop noise
  handImg, y_locArr, x_locArr = denoise(handImg)
  # crop hand
  x1, x2, y1, y2 = crop(handImg, y_locArr, x_locArr)
  
  if x2 - x1 <= 0 or y2 - y1<=0:
   continue
  for y_ in range(img.shape[0]):
   for x_ in range(img.shape[1]):
    if handImg[y_, x_] == 0:  
     img[y_, x_, 0] = 0
     img[y_, x_, 1] = 0
     img[y_, x_, 2] = 0

  HoG, HogImg = feature.hog( cv2.resize( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2], (64, 128)),
           orientations=9,
           pixels_per_cell=(8, 8), 
           cells_per_block=(2, 2), 
           block_norm="L2", 
           transform_sqrt=True, 
           visualise=True)

  XAtt.append(HoG.tolist())
  yAtt.append(rgbPath.split("/")[2])
 return XAtt, yAtt

def trainingModel(imgsFile):
 XAtt = []
 yAtt = []
 sys.setrecursionlimit(640 * 480)
 rgbImgPathList, depImgPathList = getFilePath(imgsFile)
 startTime = time.time()
 XAtt, yAtt = generateDtaset(rgbImgPathList, depImgPathList, XAtt, yAtt)
 endTime = time.time()
 print("Generating dataset costs : " + str(endTime - startTime) + " sec")
 SVC(XAtt, yAtt, 10)

def main(imgsFile, modelFile):
 
 try:
  open(modelFile, 'rb')
 except IOError:
  trainingModel(imgsFile)
 loadModel = pickle.load(open(modelFile, 'rb'))
 img = cv2.imread("WechatIMG13.jpeg")
 img = cv2.resize(img, (64, 128)) 


 rows, cols = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape
 M = cv2.getRotationMatrix2D((cols,rows),90,1)
 dst = cv2.warpAffine(img,M,(cols,rows))
 cv2.imshow('image', dst)
 cv2.waitKey(0)


 startTime = time.time()
 HoG = feature.hog( cv2.resize( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 128)),
      orientations=9,
      pixels_per_cell=(8, 8), 
         cells_per_block=(2, 2), 
         block_norm="L2", 
      transform_sqrt=True, 
      visualise=False)
 
 XTest = []
 YTest = []
 XTest.append(HoG)
 YTest.append("G9")
 result = loadModel.score(XTest, YTest)
 endTime = time.time()

 print(result, str(endTime-startTime))

main("acquisitions", "model.sav")