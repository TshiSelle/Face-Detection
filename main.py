import cv2 as cv
import time
import datetime

show = cv.VideoCapture(0)

face_det = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_det = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_fullbody_default.xml")
recording = True

frameSize = (int(show.get(3)), int(show.get(4)))
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter("Footage.mp4", fourcc, 30, frameSize)
#program loop
while True:
  _, frame = show.read()
  
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  faces = face_det.detectMultiScale(gray, 1.3, 5)
  bodies = body_det.detectMultiScale(gray, 1.3, 5)
  
  if len(faces) + len(bodies) > 0:
    recording = True
    
  out.write(frame)
  
  #for(x,y,width, height) in faces:
  #  cv.rectangle(frame, (x, y), (x + width, y + height), (130, 0, 75), 3)
  #for(x,y,width, height) in bodies:
  #  cv.rectangle(frame, (x, y), (x + width, y + height), (0, 20, 200), 3)
  
  cv.imshow("Footage", frame)
  
  if cv.waitKey(1) == ord('q'):
    break
# end of loop

out.release()
show.release()
cv.destroyAllWindows()