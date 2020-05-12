import dlib
import cv2 
import numpy as np
import sqlite3
from sqlite3 import Error
import datetime
from imutils import face_utils
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import face_recognition
import warnings
warnings.filterwarnings('ignore')


now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
recordingDate = str(day)+month


def add_user_yawn(name, tableName):
    try:
        reaction = 'yawn'
        conn = sqlite3.connect("EmployeeDatabase.db")
        insertQuery= "INSERT INTO "+str(tableName)+"(Emp_ID, yawn, YawnTimestamp) VALUES ("+str(name)+", ' "+str(reaction)+" ',datetime('now','localtime'))";
        conn.execute(insertQuery)
        conn.commit()
        conn.close()
        
    except Error as e:
        print('Timestamp could not be added')
        print(e)

def create_user_yawn(name):
    try:
        tableName_1 = 'Yawn_'+name+'_'+str(recordingDate)
        print("FLAG 1 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        createTableQuery = 'CREATE TABLE '+str(tableName_1)+' (Emp_ID BIGINT (5) REFERENCES Employee (Emp_ID), yawn TEXT NOT NULL, YawnTimestamp TEXT NOT NULL)';
        conn.execute(createTableQuery)
        conn.close()
        
        add_user_yawn(name, tableName_1)
                        
    except sqlite3.OperationalError as e:    
        print("FLAG 1 B")
        add_user_yawn(name, tableName_1)
        print(e)



dlib_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)


data = pickle.loads(open('encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX  

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def get_landmarks(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rects = dlib_detector(gray, 1)
    
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


yawns = 0
yawn_status = False 

vs = VideoStream().start()

fps = FPS().start()

while True:
    # Reduce width to increase computation power
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    rects = dlib_detector(gray, 0)
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
      
    boxes = [(b, a + c, b + d, a) for (a, b, c, d) in faces]

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
                

    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name != 'Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
          
            detected_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))

            for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)                
                    image_landmarks, lip_distance = yawn_detection(frame)
                    mouth = shape[mStart:mEnd]
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (255, 255, 255), 1)
                    prev_yawn_status = yawn_status
                
            if lip_distance > 25:
                yawn_status = True
                cv2.putText(frame, "Yawn Added", (280,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
                output_text = "Yawn Count: "+ str(yawns + 1)
                #addYawnCount(profile[0])
                create_user_yawn(name)
                
            else:
                yawn_status = False
                
            if prev_yawn_status == True and yawn_status == False:
                yawns += 1
                #time.sleep(2)
        else:
            cv2.rectangle(frame, (top-22,y-90), (top+left+22, y-22), -1)
            cv2.putText(frame, "Unknown", (top,y-40), font, .70, (0,0,0), 2)
            


    # Display the video im with the bounded rectangle
    cv2.imshow('Taking Attendance',frame) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
vs.stream.release()

# Close all windows
cv2.destroyAllWindows()
            
vs.stop() 