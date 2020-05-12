import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import datetime
import dlib
import sqlite3
from sqlite3 import Error
import face_recognition
import warnings
warnings.filterwarnings('ignore')


now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
recordingDate = str(day)+month


def add_user_sleep(name, tableName):
    try:
        reaction = 'sleep'
        conn = sqlite3.connect("EmployeeDatabase.db")
        insertQuery= "INSERT INTO "+str(tableName)+"(Emp_ID, sleep, SleepTimestamp) VALUES ("+str(name)+", ' "+str(reaction)+" ',datetime('now','localtime'))";
        conn.execute(insertQuery)
        conn.commit()
        conn.close()
        
    except Error as e:
        print('Timestamp could not be added')
        print(e)

def create_user_sleep(name):
    try:
        tableName_1 = 'Sleep_'+name+'_'+str(recordingDate)
        print("FLAG 1 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        createTableQuery = 'CREATE TABLE '+str(tableName_1)+' (Emp_ID BIGINT (5) REFERENCES Employee (Emp_ID), sleep TEXT NOT NULL, SleepTimestamp TEXT NOT NULL)';
        conn.execute(createTableQuery)
        conn.close()
        
        add_user_sleep(name, tableName_1)
                        
    except sqlite3.OperationalError as e:    
        print("FLAG 1 B")
        add_user_sleep(name, tableName_1)
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




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default = "shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="Siren.mp3", help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Eye Aspect Ratio and Frame Threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 38

# Frame counter
COUNTER = 0


print("Loading facial landmark predictor...")
dlib_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)


data = pickle.loads(open('encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
                    cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink im counter
                    if ear < EYE_AR_THRESH:

                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            #addSleepCount(str(profile[0]))
                            #print("Sleep Count for:"+str(profile[0]))
                            create_user_sleep(name)
                            cv2.putText(frame, "SLEEP COUNT ADDED FOR:"+name, (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            		   # otherwise, the eye aspect ratio is not below the blink
                    # threshold, so reset the counter and alarm
                    else:
                        COUNTER = 0

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)


        else:
            #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), -1)
            cv2.putText(frame, "Unknown", (top,y-40), cv2.FONT_HERSHEY_SIMPLEX, .70, (0,0,0), 2)


        # Put text describe who is in the picture
        #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,255,255), -1)
        #cv2.putText(im, str(Id), (x,y-40), font, 1, (0,0,0), 3)

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
