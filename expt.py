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
    fromaddr = "droid7developer@gmail.com"
    toaddr = "vijaypalsingh.dhaka@jaipur.manipal.edu"
       
    # instance of MIMEMultipart 
    msg = MIMEMultipart() 
      
    # storing the senders email address   
    msg['From'] = fromaddr 
      
    # storing the receivers email address  
    msg['To'] = toaddr 
      
    # storing the subject  
    msg['Subject'] = "Intruder Detected"
      
    # string to store the body of the mail 
    body = "Dear User, an intruder has been detected at "+detected_at+'. Please check the attached image to verify.\nStay Safe, \nAWAKE.AI'
      
    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 
      
    # open the file to be sent  
    filename = img_name
    attachment = open("/home/pi/Desktop/Awake/"+filename, "rb") 
      
    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 
      
    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 
      
    # encode into base64 
    encoders.encode_base64(p) 
       
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
      
    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 
      
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
      
    # start TLS for security 
    s.starttls() 
      
    # Authentication 
    s.login(fromaddr, "17111996rjt") 
      
    # Converts the Multipart msg into a string 
    text = msg.as_string() 
      
    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 
      
    # terminating the session 
    s.quit() 

    
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    faces = detector.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        # Create rectangle around the face
        cv2.rectangle(frame, (x-20,y-20), (x+w+20,y+h+20), (0,0,0), 4)
        detected_at = time.ctime()
        img_name = 'intruder_detected_at: '+str(detected_at)+'.png'
        cv2.imwrite(img_name, gray[y:y+h,x:x+w])
        
        send_email(img_name, detected_at)
        time.sleep(5)
 
 
 
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()