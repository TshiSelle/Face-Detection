import cv2 as cv
import time
import datetime

show = cv.VideoCapture(0)

face_det = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
while True:
  _, frame = show.read()
  cv.imshow("Footage", frame)
  
  if cv.waitKey(1) == ord('q'):
    break
  
show.release()
cv.destroyAllWindows()