import numpy as np
import cv2
from keras.preprocessing import image
import sqlite3
from sqlite3 import Error
import datetime
        
def assure_reaction_database_exists(tableName):
    try:
        #print("FLAG 1 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        createTableQuery = "CREATE TABLE "+str(tableName)+" (Emp_ID BIGINT (5) PRIMARY KEY REFERENCES Employee (Emp_ID) UNIQUE, angry BIGINT (1000) DEFAULT (0), disgust  BIGINT (1000) DEFAULT (0), fear BIGINT (1000) DEFAULT (0), happy BIGINT (1000) DEFAULT (0), sad BIGINT (1000) DEFAULT (0),  surprise BIGINT (1000) DEFAULT (0), neutral  BIGINT (1000) DEFAULT (0))";
        conn.execute(createTableQuery)
        conn.close()
        
    except sqlite3.OperationalError as e:
        print("FLAG 1 B")
        print(e)

now = datetime.datetime.now()
day = now.day
month = now.strftime("%B")
year = now.year
recordingDate = str(day)+str(month)+str(year)

reactionTableName = recordingDate + '_Reaction'

assure_reaction_database_exists(reactionTableName)
        
def getProfile(id):
    try:
        #print("FLAG 2 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        selectQuery = 'SELECT * FROM Employee WHERE Emp_ID='+str(id)
        cursor = conn.execute(selectQuery)
        profile = 'None'
        for row in cursor:
            profile = row
        conn.close()
        return profile
    
    except Error as e:
        print("FLAG 2 B")
        print(e)


def add_reaction(id,reaction):
    try:
        #print("FLAG 3 A")
        conn = sqlite3.connect('EmployeeDatabase.db')
        insertQuery = "INSERT INTO "+str(reactionTableName)+"(Emp_ID,"+str(reaction)+") VALUES ("+str(id)+", 0)"
        conn.execute(insertQuery)
        conn.commit()
        conn.close()
        
    except Error as e:
        #print("FLAG 3 B")
        conn = sqlite3.connect('EmployeeDatabase.db')
        updateReactionQuery = "UPDATE "+str(reactionTableName)+" SET "+str(reaction)+ "  = "+str(reaction)+ " + 1 WHERE Emp_ID ="+str(id)
        conn.execute(updateReactionQuery)
        conn.commit()
        conn.close()
        
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


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

cam = cv2.VideoCapture(0)
while(True):
    
    ret , im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, 1.2,5)


    for (x,y,w,h) in faces:
        # Read the video im
        ret, im =cam.read()

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
                    #cv2.putText(im, str(profile[2]), (x,y-20), font, .70, (0,0,0), 2)
                    detected_face = im[int(y):int(y+h), int(x):int(x+w)]
                    
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    detected_face = cv2.resize(detected_face, (48, 48))
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 255
                    predictions = model.predict(img_pixels)
                	   #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                    max_index = np.argmax(predictions[0])
                    emotion = emotions[max_index]
                    cv2.putText(im, emotion, (330,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
                    #print(emotion)
                    add_reaction(id,emotion)
            
            else:
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), -1)
                cv2.putText(im, "Unknown", (x,y-40), font, .70, (0,0,0), 2)
            
            
        
    # Display the video im with the bounded rectangle
    cv2.imshow('Recognizing Emotions',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
            