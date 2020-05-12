# Import OpenCV2 for image processing
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import datetime
import dlib
import sqlite3
from sqlite3 import Error
import ctypes
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

'''
def assure_database_exists(tableName):
    try:
        #conn = sqlite3.connect("StudentDatabase.db")
        #existsQuery = "IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'TheSchema' AND  TABLE_NAME = '"+str(subjectCode)+"') BEGIN END"

        #insertQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now'))"
        #insertQuery = "INSERT INTO"+str(subjectCode)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("
        #conn.execute(existsQuery)
        #conn.close()

        conn = sqlite3.connect("StudentDatabase.db")
        createTableQuery = "CREATE TABLE "+str(tableName)+"(ID BIGINT (9) REFERENCES Student(ID) NOT NULL UNIQUE ON CONFLICT REPLACE, SubjectCode VARCHAR NOT NULL, AttendanceTimestamp TEXT NOT NULL)"
        #createTableQuery = "CREATE TABLE "+str(subjectCode)+"(ID BIGINT (9) REFERENCES Student (ID) NOT NULL UNIQUE ON CONFLICT REPLACE, SubjectCode VARCHAR NOT NULL, AttendanceTimestamp TEXT NOT NULL)"

        conn.execute(createTableQuery)
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database", "ERROR !", 1)
        #print(e)

    except sqlite3.OperationalError as e:
        #conn = sqlite3.connect("StudentDatabase.db")
        #createTableQuery = "CREATE TABLE "+str(subjectCode)+"(ID BIGINT (9) REFERENCES Student(ID) NOT NULL UNIQUE ON CONFLICT REPLACE, SubjectCode VARCHAR NOT NULL, AttendanceTimestamp TEXT NOT NULL)"
        #conn.execute(createTableQuery)
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database assure_database_exits", "ERROR !", 1)
        print(e)
'''
def assure_sleep_database_exists(tableName):
    try:
        conn = sqlite3.connect("StudentDatabase.db")
        createSleepTableQuery = "CREATE TABLE "+str(tableName)+"(ID BIGINT (9) PRIMARY KEY REFERENCES Student (ID) ON UPDATE NO ACTION, SleepTimeInSecs BIGINT DEFAULT (0))"
        conn.execute(createSleepTableQuery)

    except sqlite3.OperationalError as e:
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database assure_database_exits", "ERROR !", 1)
        print(e)


subjectCode = input('Enter Subject Code: ')
now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
lectureDate = str(day)+month+str(year)
#print('Preparing Attendance Table For: ',lectureDate)
#attendanceTableName = subjectCode +'_'+ str(lectureDate)
#print('Table Name: ',attendanceTableName)
#assure_database_exists(attendanceTableName)

sleepTableName = subjectCode+'_'+'SleepCount'
print('Preparing Sleep Table For: ', subjectCode)
print('Table Name: ',sleepTableName)
assure_sleep_database_exists(sleepTableName)



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

def addTimestamp(id):
    try:
        conn = sqlite3.connect("StudentDatabase.db")
        #insertTimestampQuery = "INSERT INTO Attendance date('now') WHERE ID="+str(id)
        #insertQuery= "INSERT INTO Attendance(ID,AttendanceTimestamp) VALUES ("+str(id)+", datetime('now'))"

        #insertQuery= "INSERT INTO "+str(subjectCode)+"(ID,SubjectCode,AttendanceTimestamp,SleepCount) VALUES ("+str(id)+", ' "+str(subjectCode)+" ', datetime('now','localtime'),"+str(SleepCount)+")"
        #insertIntoAttendanceQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp,SleepCount) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now','localtime'),"+str(SleepCount)+")"

        insertQuery= "INSERT INTO "+str(sleepTableName)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now','localtime'))"
        #insertIntoAttendanceQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now','localtime'))"

        #insertQuery= "INSERT INTO "+str(subjectCode)+"(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',DATEADD(360,getdate()))"

        #insertQuery= "INSERT INTO Attendance(ID,SubjectCode,AttendanceTimestamp) VALUES ("+str(id)+", ' "+str(subjectCode)+" ',datetime('now'))"
        conn.execute(insertQuery)
        #conn.execute(insertIntoAttendanceQuery)
        #conn.execute("INSERT INTO Attendance(ID,AttendanceTimestamp) VALUES ("+str(id)+",' datetime('now')')")
        #existsQuery="INSERT INTO Student(RegistrationNumber,Name,Branch) Values("+str(RegistrationNumber)+",' "+str(Name)+" ',' "+str(Branch)+" ')"
        conn.commit()
        conn.close()
    except Error as e:
        ctypes.windll.user32.MessageBoxW(0, "Timestamp could not be added", "DATABSE ERROR !", 1)
        print(e)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear



def addSleepCount(id):

    try:
        insertQuery= "INSERT INTO "+str(sleepTableName)+"(ID) VALUES ("+str(id)+") ON CONFLICT(ID) DO UPDATE SET SleepTimeInSecs = SleepTimeInSecs + 1"
        #insertQuery= "INSERT INTO "+str(sleepTableName)+"(ID,SleepTimeInSecs) VALUES ("+str(id)+", 0)"
        conn = sqlite3.connect("StudentDatabase.db")
        insertQuery= "INSERT INTO "+str(sleepTableName)+"(ID,SleepTimeInSecs) VALUES ("+str(id)+", 0)"
        #updateSleepQuery = "UPDATE "+str(subjectCode)+" SET SleepCount = SleepCount + 1 WHERE ID ="+id
        conn.execute(insertQuery)
        #conn.execute(updateSleepQuery)
        conn.commit()
        conn.close()

    except Error as e:
        #print("Sleep Count in EXCEPT STATEMENT")
        updateSleepQuery = "UPDATE "+str(sleepTableName)+" SET SleepTimeInSecs = SleepTimeInSecs + 1 WHERE ID ="+id
        #ctypes.windll.user32.MessageBoxW(0, "SleepCount could not be added", "DATABSE ERROR !", 1)
        conn.execute(updateSleepQuery)
        conn.commit()
        conn.close()
        #print(e)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default = "shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="Siren.mp3", help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# ims the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 48

# initialize the im counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video im capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video im
    ret, im =cam.read()

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

                #addTimestamp(str(profile[0]))

                for rect in rects:


                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(im, [leftEyeHull], -1, (255, 255, 255), 1)
                    cv2.drawContours(im, [rightEyeHull], -1, (255, 255, 255), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink im counter
                    if ear < EYE_AR_THRESH:

                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            addSleepCount(str(profile[0]))
                            print("Sleep Count for:"+str(profile[0]))
                            cv2.putText(im, "SLEEP COUNT ADDED FOR:"+str(profile[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            		   # otherwise, the eye aspect ratio is not below the blink
                    # threshold, so reset the counter and alarm
                    else:
                        COUNTER = 0

                    cv2.putText(im, "EAR: {:.2f}".format(ear), (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)


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
cam.release()

# Close all windows
cv2.destroyAllWindows()