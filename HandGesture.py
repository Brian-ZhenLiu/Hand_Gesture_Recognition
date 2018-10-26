import cv2
import numpy as np
import hogsvm2
import pickle
import time
from skimage import feature

def HOG(img, imgResize=(64, 128), bin=9, cell=(8, 8), block=(2, 2), norm="L2", sqrt=True, visualize=False):
 if visualize == False:
  hog = feature.hog( cv2.resize( img, imgResize),
      orientations=bin,
      pixels_per_cell=cell, 
         cells_per_block=block, 
         block_norm=norm, 
      transform_sqrt=sqrt, 
      visualise=visualize)
  return hog
 else:
  hog, hogImg = feature.hog( cv2.resize( img, imgResize),
      orientations=bin,
      pixels_per_cell=cell, 
         cells_per_block=block, 
         block_norm=norm, 
      transform_sqrt=sqrt, 
      visualise=visualize)
  return hog, hogImg

def trackHand(img, loadModel):
 YTest = [["G1"],["G2"],["G3"],["G4"],["G5"],["G6"],["G7"],["G8"],["G9"],["G10"]]
 startTime = time.time()
 grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 cropWindow = grayImg[0:64*4, 0:128*4]
 mList = []
 for i in range(len(YTest)):
  result = loadModel.score([HOG(cropWindow)], YTest[i])
  if result != 0:
   return YTest[i]

def text2Num(text):
  num = 0
  if text == 'G1':
    num = 1
  if text == 'G2':
    num = 2
  if text == 'G3':
    num = 3
  if text == 'G4':
    num = 4 
  if text == 'G5':
    num = 5
  if text == 'G6':
    num = 6
  if text == 'G7':
    num = 7
  if text == 'G8':
    num = 8
  if text == 'G9':
    num = 9
  if text == 'G10':
    num = 10
  return num


def show_webcam(mirror=True):
 try:
  open("svc.sav", 'rb')
 except IOError:
  tranModel = handGestureModelTraining("acquisitions")
 loadModel = pickle.load(open("svc.sav", 'rb'))

 cam = cv2.VideoCapture(0)

 startTime = time.time()
 while True:

  endTime = time.time()
  if endTime - startTime >= 0.25:
   startTime = time.time()
  else:
   continue
  ret_val, img = cam.read()
  img = np.float32(img) / 255.0
  if mirror:
   img = cv2.flip(img, 1)
  (x, y, u, v) = (0, 0, 64*4+200, 128*4)
  
  cv2.rectangle(img, (x, y), (x + u, y + v), (255, 0, 0), 2)
  
  text = str(trackHand(img, loadModel)[0])
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img, str(text2Num(text)), (64*4+900, 100), font, 4, (255,255,255), 2, cv2.LINE_AA)

  cv2.imshow('NormalCam', img)
  cv2.moveWindow('NormalCam', 0, 0)
  if cv2.waitKey(1) == 27: 
   break
 cv2.destroyAllWindows()

def main():
 show_webcam()

main()
