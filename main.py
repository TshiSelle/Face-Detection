import cv2 as cv
import time
import datetime

show = cv.VideoCapture(0)

while True:
  _, frame = show.read()
  cv.imshow("Footage", frame)
  