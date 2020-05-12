import cv2
import os
import sqlite3
from sqlite3 import Error

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def AddUpdateUser(Emp_ID, Emp_Name, Emp_Grade, Emp_Team, Task_Skills_Req, Emp_Skills, Email_Id):
    try:
        conn= sqlite3.connect("EmployeeDatabase.db")
        #existsQuery = "INSERT INTO Student(ID,Name,Branch,Gender) Values("+str(ID)+",' "+str(Name)+" ',' "+str(Branch)+" ',' "+str(Gender)+" ')"
        existsQuery = "INSERT INTO Employee(Emp_ID, Emp_Name, Emp_Grade, Emp_Team, Task_Skills_Req, Emp_Skills, Email_Id) Values ("+str(Emp_ID)+",' "+str(Emp_Name)+" ',' "+str(Emp_Grade)+" ',' "+str(Emp_Team)+" ',' "+str(Task_Skills_Req)+" ',' "+str(Emp_Skills)+" ',' "+str(Email_Id)+" ')"
        conn.execute(existsQuery)
        conn.commit()
        conn.close()
    except Error as e:
        #ctypes.windll.user32.MessageBoxW(0, "Could Not Connect to Database", "ERROR !", 1)
        print('Error: ',e)
        

# Enter the data
Emp_ID= input('Enter Employee ID: ')
Emp_Name = input('\n'+'Enter Employee Full Name: ')
Emp_Grade = input('\n'+'Enter Employee Grade: ')
Emp_Team = input('\n'+'Enter Employee\'s Team Name: ')
Task_Skills_Req = input('\n'+'Enter Task Skills Required: ')
Emp_Skills = input('\n'+'Enter Employee Skills: ')
Email_Id = input('\n'+'Enter Employee Email Id: ')

AddUpdateUser(Emp_ID, Emp_Name, Emp_Grade, Emp_Team, Task_Skills_Req, Emp_Skills, Email_Id)
count = 0

assure_path_exists("dataset/")

while(True):
    _, image_frame = vid_cam.read()

    #IMAGE AUGMENTATION
    
    #1. Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    
    #2. Rotation
    #rotate = imutils.rotate(gray, 15)

    #3. Median Blur
    #blur = cv2.blur(gray,(70,70))
    #blur = cv2.medianBlur(gray, 5)
    
    #4. Vertical Axis Flip
#    flipped = cv2.flip(gray,1)
    
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        if count > 20:    
            #ctypes.windll.user32.MessageBoxW(0, "Images have been added to database", "DONE !", 1)
            #print('Images have been added to database', flush = True)
                
            break 
        else:            
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
            count = count + 1
            cv2.imwrite("dataset/User." + str(Emp_ID) + '.' + str(count) + ".png", gray[y:y+h,x:x+w])
    
            #count = count + 1
            #cv2.imwrite("dataset/User." + str(id) + '.' + str(count) + ".png", rotate[y:y+h,x:x+w])
    
            
            #count = count + 1
            #cv2.imwrite("dataset/User." + str(id) + '.' + str(count) + ".jpg", blur[y:y+h,x:x+w])
            #cv2.imwrite("dataset/User." + str(id) + '.' + str(count) + ".jpg", flipped[y:y+h,x:x+w])
            
            
            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('Taking Images', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


vid_cam.release()

cv2.destroyAllWindows()