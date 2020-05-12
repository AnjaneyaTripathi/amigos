import cv2
import imutils
import time
from imutils.video import VideoStream

    
vs = VideoStream().start()

time.sleep(2.0)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3,5)

    for(x,y,w,h) in faces:

        cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,0,0), 4)

        detected_at = time.ctime()

        img_name = 'intruder_detected_at: '+str(detected_at)+'.png'

        cv2.imwrite(img_name, gray[y:y+h,x:x+w])

        
    cv2.imshow("Intruder Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

vs.stop()

