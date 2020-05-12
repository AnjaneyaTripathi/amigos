import dlib
import cv2 
import numpy as np
import sqlite3
from sqlite3 import Error
import ctypes
import time
from imutils import face_utils

def assure_yawn_database_exists(tableName):
    try:
        conn = sqlite3.connect("StudentDatabase.db")
        createSleepTableQuery = "CREATE TABLE "+str(tableName)+"(ID BIGINT (9) PRIMARY KEY REFERENCES Student (ID) ON UPDATE NO ACTION, YawnTimeInSecs BIGINT DEFAULT (0))"
        conn.execute(createSleepTableQuery)

    except sqlite3.OperationalError as e:
        print(e)  
        
subjectCode = input('Enter Subject Code: ')
yawnTableName = subjectCode + "_YawnCount"
assure_yawn_database_exists(yawnTableName)

def getProfile(id):
    try:
        conn = sqlite3.connect("StudentDatabase.db")
        selectQuery = "SELECT * FROM Student WHERE ID="+str(id)
        cursor = conn.execute(selectQuery)
        profile = 'None'
        for row in cursor:
            profile = row
        conn.close()
        return profile
    
    except Error as e:
        ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database getProfile", "ERROR !", 1)
        print(e)

def addYawnCount(id):
    try:
        #print("TRY statement in addYawnCount")        
        insertQuery= "INSERT INTO "+str(yawnTableName)+"(ID,YawnTimeInSecs) VALUES ("+str(id)+", 0)"
        conn = sqlite3.connect("StudentDatabase.db")
        conn.execute(insertQuery)
        conn.commit()
        conn.close()

    except Error as e:
        #print("EXCEPT statement in addYawnCount")
        updateYawnQuery = "UPDATE "+str(yawnTableName)+" SET YawnTimeInSecs = YawnTimeInSecs + 1 WHERE ID ="+str(id)
        #ctypes.windll.user32.MessageBoxW(0, "SleepCount could not be added", "DATABSE ERROR !", 1)
        conn.execute(updateYawnQuery)
        conn.commit()
        conn.close()
        #print(e)

        
        
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX  

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0], point[0,1])
        cv2.putText(im, str(idx), pos,
                    fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale = 0.4, color = (0,0,255))
        cv2.circle(im,pos,3,color = (0,0,255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis = 0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis = 0)
    return int(bottom_lip_mean[:,1])

def yawn_detection(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
                  
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance
    #return  lip_distance



cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True:
    ret, im = cap.read()
    
    # Convert the captured im into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # Get all face from the video im
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,0,0), 4)

        # Recognize the face belongs to which ID
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
         
        profile = getProfile(id)
        if confidence < 100:
            profile = getProfile(id)
            if profile != None:
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), -1)
                cv2.putText(im, str(profile[0]), (x,y-60), font, .70, (255,255,255), 2)
                cv2.putText(im, str(profile[1]), (x,y-40), font, .70, (255,255,255), 2)
                #cv2.putText(im, str(profile[2]), (x,y-20), font, .70, (255,255,255), 2)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)                
                    image_landmarks, lip_distance = yawn_detection(im)
                    mouth = shape[mStart:mEnd]
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(im, [mouthHull], -1, (255, 255, 255), 1)
                    prev_yawn_status = yawn_status
                
                if lip_distance > 25:
                    yawn_status = True
                    cv2.putText(im, "Yawn Added", (280,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
                    output_text = "Yawn Count: "+ str(yawns + 1)
                    addYawnCount(profile[0])
                    
                else:
                    yawn_status = False
                    
                if prev_yawn_status == True and yawn_status == False:
                    yawns += 1
                    #time.sleep(2)
        else:
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), -1)
            cv2.putText(im, "Unknown", (x,y-40), font, .70, (0,0,0), 2)
            
            
        # Put text describe who is in the picture
        #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,255,255), -1)
        #cv2.putText(im, str(Id), (x,y-40), font, 1, (0,0,0), 3)

    # Display the video im with the bounded rectangle
    cv2.imshow('Taking Attendance',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
    