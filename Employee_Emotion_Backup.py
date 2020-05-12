from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import cv2
import numpy as np
from keras.preprocessing import image
import sqlite3
from sqlite3 import Error
import datetime
import warnings
warnings.filterwarnings('ignore')
        
def assure_reaction_database_exists(tableName):
    try:
        conn = sqlite3.connect('EmployeeDatabase.db')
        createTableQuery = 'CREATE TABLE '+str(tableName)+' (Emp_ID BIGINT (5) PRIMARY KEY REFERENCES Employee (Emp_ID) UNIQUE, angry BIGINT (1000) DEFAULT (0), disgust  BIGINT (1000) DEFAULT (0), fear BIGINT (1000) DEFAULT (0), happy BIGINT (1000) DEFAULT (0), sad BIGINT (1000) DEFAULT (0),  surprise BIGINT (1000) DEFAULT (0), neutral  BIGINT (1000) DEFAULT (0))';
        conn.execute(createTableQuery)
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(e)

now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
recordingDate = str(day)+month


reactionTableName = 'Reaction_' + str(recordingDate) 

assure_reaction_database_exists(reactionTableName)


def add_reaction(id,reaction):
    try:
        conn = sqlite3.connect('EmployeeDatabase.db')
        insertQuery = "INSERT INTO "+str(reactionTableName)+"(Emp_ID,"+str(reaction)+")     VALUES ("+str(id)+", 0)"
        conn.execute(insertQuery)
        conn.commit()
        conn.close()
        
    except Error as e:
        conn = sqlite3.connect('EmployeeDatabase.db')
        updateReactionQuery = "UPDATE "+str(reactionTableName)+" SET "+str(reaction)+ " = "+str(reaction)+ " + .5 WHERE Emp_ID ="+str(id)
        conn.execute(updateReactionQuery)
        conn.commit()
        conn.close()


def add_user_reaction(name, reaction, tableName):
    try:
        
        conn = sqlite3.connect("EmployeeDatabase.db")
        insertQuery= "INSERT INTO "+str(tableName)+"(Emp_ID, emotion, AttendanceTimestamp) VALUES ("+str(name)+", ' "+str(reaction)+" ',datetime('now','localtime'))";
        conn.execute(insertQuery)
        conn.commit()
        conn.close()
        
    except Error as e:
        print('Timestamp could not be added')
        print(e)



def create_user_reaction(name, reaction):
    try:
        tableName_1 = 'Reaction_'+name+'_'+str(recordingDate)
        print("FLAG 1 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        createTableQuery = 'CREATE TABLE '+str(tableName_1)+' (Emp_ID BIGINT (5) REFERENCES Employee (Emp_ID), emotion BIGINT (1000) DEFAULT (0), AttendanceTimestamp TEXT NOT NULL)';
           
        conn.execute(createTableQuery)
        conn.close()
        
        add_user_reaction(name, reaction, tableName_1)
                        
    except sqlite3.OperationalError as e:    
        print("FLAG 1 B")
        add_user_reaction(name, reaction, tableName_1)
        print(e)


from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')


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
                

    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name != 'Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
          
            detected_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)
        	   #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
            emotion = emotions[max_index]
            cv2.putText(frame, emotion, (330,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
            #print(emotion)
            add_reaction(name,emotion)
            
            create_user_reaction(name,emotion)
            

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Stop the camera
vs.stream.release()

# Close all windows
cv2.destroyAllWindows()
            
vs.stop()