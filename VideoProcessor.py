# -*- coding: utf-8 -*-
import cv2 as cv
from LaneMarkersModel import LaneMarkersModel
from LaneMarkersModel import normalize
import numpy as np
from Sensor import LaneSensor
from LineDetector import LineDetector

from optparse import OptionParser 

# Разбор опций командной строки
parser = OptionParser() 
parser.add_option("-i", "--input", dest="input_video", help="", default="video/in/video_in.mp4")  
parser.add_option("-o", "--output", dest="output_video", help="", default="video/out/out_video.mp4")  
(options, args) = parser.parse_args() 

# Входной поток видео
stream = cv.VideoCapture(options.input_video) #6 7 8
if stream.isOpened() == False:
    print "Cannot open input video"
    exit()

# Выходной поток видео
videoWriter = cv.VideoWriter(options.output_video, cv.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720), 1)

# размер области захвата видео для определения полос
# cropArea = [x1, y1, x2, y2]
#
# (0,0)|
#  ----|----------------------->
#      |  (x1,y1)
#      | *____________ 
#      | |            |
#      | |            |
#      | |            |
#      | |            |
#      | |            |
#      | |            |
#      | -------------* (x2,y2)
#      |
#
# cropArea = [0, 124, 637, 298] # для видео 640*480
cropArea = [0, 250, 1240, 645]  # для видео 1280*720
sensorsNumber = 50
sensorsWidth = 70

# #6L
# line1LStart = np.array([35, 128])
# line1LEnd = np.array([220, 32])
#
# #6R
# line1RStart = np.array([632, 146])
# line1REnd = np.array([476, 11])

# расположение синих линий на видео в координатах
#L
line1LStart = np.array([35, 300])
line1LEnd = np.array([220, 100])

#R
line1RStart = np.array([632, 300])
line1REnd = np.array([476, 100])

#get first frame for color model
# flag, imgFull = stream.read()
# img = imgFull[cropArea[0]:cropArea[2], cropArea[1]:cropArea[3]]
# img = imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]

#Initialize left lane
leftLineColorModel = LaneMarkersModel()
#leftLineColorModel.InitializeFromImage(np.float32(img)/255.0, "Select left line")
leftLine = LineDetector(cropArea, sensorsNumber, sensorsWidth, line1LStart, line1LEnd, leftLineColorModel)

#Initialize right lane
rightLineColorModel = LaneMarkersModel()
#rightLineColorModel.InitializeFromImage(np.float32(img)/255.0, "Select right line")
rightLine = LineDetector(cropArea, sensorsNumber, sensorsWidth, line1RStart, line1REnd, rightLineColorModel)

frameNumber = 0
while(cv.waitKey(1) != 27): # пока не нажат esc
    frameNumber+=1
    print frameNumber
    
    # захватываем текущий кадр и кладем его в переменную img
    flag, imgFull = stream.read()
    if flag == False: break #end of video

    #do some preprocessing to share results later
    img = np.float32(imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]])/255.0
   
    # меняем цветовую модель на HSV
    hsv = np.float32(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    # hsv = cv.GaussianBlur(hsv, (25, 25), 2)
 
    canny = cv.Canny(cv.cvtColor(np.uint8(img*255), cv.COLOR_RGB2GRAY), 0, 170)
    # canny = cv.Canny(cv.cvtColor(np.uint8(img*255), cv.COLOR_RGB2GRAY), 70, 170)
 
    #make output images
    outputImg = img.copy()
    outputFull = imgFull.copy()
    outputHSV = hsv.copy()
    outputCANNY = canny.copy()

    #process frame
    leftLine.ProcessFrame(img, hsv, canny, outputImg, outputFull)
    rightLine.ProcessFrame(img, hsv, canny, outputImg, outputFull)
    
    #show output
    cv.imshow("Output", outputImg)
    cv.imshow("Output full", outputFull)
    # cv.imshow("Output hsv", outputHSV)
    # cv.imshow("Output canny", outputCANNY)
    
    #write output
    videoWriter.write(outputFull)
    
cv.destroyAllWindows()
