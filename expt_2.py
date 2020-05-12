import cv2
import imutils
import time
from imutils.video import VideoStream
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

    body = "Dear User, an intruder has been detected at "+detected_at+'. Please check the attached image to verify.\nStay Safe, \nAWAKE.AI'

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

    
vs = VideoStream(usePiCamera=True).start()

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

        send_email(img_name, detected_at)

        time.sleep(5)

    cv2.imshow("Intruder Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

vs.stop()

