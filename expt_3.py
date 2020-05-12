from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 

def send_email(img_name, detected_at):
    from_email = "droid7developer@gmail.com"

    #to_email = "vijaypalsingh.dhaka@jaipur.manipal.edu"
    to_email = "rjtmehta99@gmail.com"
    
    mail = MIMEMultipart() 

    mail['From'] = from_email 

    mail['To'] = to_email 

    mail['Subject'] = "Intruder Detected"

    body = "Dear User, an intruder has been detected at "+detected_at+'. Please check the attached image to verify.\nStay Safe, \nRajat Mehta'

    mail.attach(MIMEText(body, 'plain')) 

    filename = img_name
    
    attachment = open("/home/pi/Desktop/Awake/"+filename, "rb") 

    mime_ins = MIMEBase('application', 'octet-stream') 
    
    mime_ins.set_payload((attachment).read()) 
    
    encoders.encode_base64(mime_ins) 
    
    mime_ins.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
    
    mail.attach(mime_ins) 
    
    sess = smtplib.SMTP('smtp.gmail.com', 587) 
    
    sess.starttls() 
    
    sess.login(from_email, "17111996rjt") 
    
    text = mail.as_string() 
    
    sess.sendmail(from_email, to_email, text) 
    
    sess.quit() 
    time.sleep(5.0)



data = pickle.loads(open('encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

fps = FPS().start()

while True:
    # Reduce width to increase computation power
    frame = vs.read()
    frame = imutils.resize(frame, width=200)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    boxes = [(b, a + c, b + d, a) for (a, b, c, d) in rects]

    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, m) in enumerate(matches) if m]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
        names.append(name)
                

    #    cv2.putText
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name == 'Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
            
            y = top - 15 if top - 15 > 15 else top + 15
            
            cv2.putText(frame, 'Intruder', (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            detected_at = time.ctime()
            
            img_name = 'intruder_detected_at: '+str(detected_at)+'.png'
            
            cv2.imwrite(img_name,frame)
            
            send_email(img_name, detected_at)


        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()