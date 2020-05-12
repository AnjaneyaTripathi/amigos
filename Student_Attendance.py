# Import OpenCV2 for image processing
import cv2
import datetime
# Import numpy for matrices calculations
#import numpy as np
import time
import sqlite3
from sqlite3 import Error

#Import ctypes for error popup generation
import ctypes

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def assure_database_exists(tableName):
    try:
        '''conn = sqlite3.connect("StudentDatabase.db")
        existsQuery = "IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'TheSchema' AND  TABLE_NAME = '"+str(subjectCode)+"') BEGIN END"

        #insertQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now'))"
        #insertQuery = "INSERT INTO"+str(subjectCode)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("
        conn.execute(existsQuery)
        conn.close()
        '''
        conn = sqlite3.connect("StudentDatabase.db")
        createTableQuery = "CREATE TABLE "+str(tableName)+"(ID BIGINT (9) REFERENCES Student(ID) NOT NULL UNIQUE ON CONFLICT REPLACE, SubjectCode VARCHAR NOT NULL, AttendanceTimestamp TEXT NOT NULL)"
        conn.execute(createTableQuery)
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database", "ERROR !", 1)
        #print(e)

    except sqlite3.OperationalError as e:
        #conn = sqlite3.connect("StudentDatabase.db")
        #createTableQuery = "CREATE TABLE "+str(subjectCode)+"(ID BIGINT (9) REFERENCES Student(ID) NOT NULL UNIQUE ON CONFLICT REPLACE, SubjectCode VARCHAR NOT NULL, AttendanceTimestamp TEXT NOT NULL)"
        #conn.execute(createTableQuery)
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database", "ERROR !", 1)
        print(e)


subjectCode = input('Enter Subject Code: ')
now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
lectureDate = str(day)+ month + str(year)
print('Preparing Attendance Table For: ',lectureDate)

attendanceTableName = subjectCode +'_'+ str(lectureDate)
print('Table Name: ',attendanceTableName)
assure_database_exists(attendanceTableName)

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
        ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database", "ERROR !", 1)
        print(e)

def addTimestamp(id):
    try:
        conn = sqlite3.connect("StudentDatabase.db")
        #insertTimestampQuery = "INSERT INTO Attendance date('now') WHERE ID="+str(id)
        #insertQuery= "INSERT INTO Attendance(ID,AttendanceTimestamp) VALUES ("+str(id)+", datetime('now'))"

        insertQuery= "INSERT INTO "+str(attendanceTableName)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now','localtime'))"
        insertIntoAttendanceQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now','localtime'))"

        #insertQuery= "INSERT INTO "+str(subjectCode)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',DATEADD(360,getdate()))"

        #insertQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now'))"
        conn.execute(insertQuery)
        conn.execute(insertIntoAttendanceQuery)
        #conn.execute("INSERT INTO Attendance(ID,AttendanceTimestamp) VALUES ("+str(id)+",' datetime('now')')")
        #existsQuery="INSERT INTO Student(RegistrationNumber,Name,Branch) Values("+str(RegistrationNumber)+",' "+str(Name)+" ',' "+str(Branch)+" ')"
        conn.commit()
        conn.close()
    except Error as e:
        ctypes.windll.user32.MessageBoxW(0, "Timestamp could not be added", "DATABSE ERROR !", 1)
        print(e)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer1/")

# Load the trained mode
recognizer.read('trainer1/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,3)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w,y+h), (0,0,0), 4)

        # Recognize the face belongs to which ID
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        profile = getProfile(id)
        if confidence < 55:
            if profile != None:
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y+h+22), -1)
                cv2.putText(im, str(profile[0]), (x,y-60), font, .70, (255,255,255), 2)
                cv2.putText(im, str(profile[1]), (x,y-40), font, .70, (255,255,255), 2)
                    #cv2.putText(im, str(profile[2]), (x,y-20), font, .70, (0,0,0), 2)
                confidence = "  {0}%".format(round(100 - confidence))
                #cv2.putText(im, confidence, (x+20,y+40), font, .70, (0,0,0), 2)
                #time.sleep(.5)
                addTimestamp(str(profile[0]))

            #else:
             #   cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,255,255), -1)
              #  cv2.putText(im, "NOT known", (x,y-40), font, 1, (0,0,0), 3)

        else:
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), -1)
            cv2.putText(im, "Unknown", (x,y-40), font, .70, (0,0,0), 2)


        # Put text describe who is in the picture
        #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,255,255), -1)
        #cv2.putText(im, str(Id), (x,y-40), font, 1, (0,0,0), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('Taking Attendance',im)

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()