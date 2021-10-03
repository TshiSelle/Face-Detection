import cv2 as cv
import time
import datetime

show = cv.VideoCapture(0)

face_det = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_det = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody_default.xml")

#program loop
while True:
  _, frame = show.read()
  
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  faces = face_det.detectMultiScale(gray, 1.3, 5)
  
  cv.imshow("Footage", frame)
  
  if cv.waitKey(1) == ord('q'):
    break
# end of loop

show.release()
cv.destroyAllWindows()